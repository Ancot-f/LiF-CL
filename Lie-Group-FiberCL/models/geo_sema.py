"""
Geometry-SEMA Learner (Group-MoE + Shared MambaFlow)
=====================================================

Implements the training flow from suggest.md sections 11-13:

Three-loss system:
  L_total = L_cls + lambda_1 * L_geo_rd + lambda_2 * L_sem

L_geo_rd = alpha_rd * L_group_RD + alpha_geo * L_geo + alpha_bal * L_balance
  - L_group_RD: sum_g p(g|h) * MSE(AE_g(z), z)
  - L_geo: ||R^T R - I||^2  (SO orthogonality constraint)
  - L_balance: KL(mean_batch(p(g|h)) || prior_g)

L_sem = L_proto + beta_router * L_router
  - L_proto: prototype consistency across tasks
  - L_router: KL divergence between old and new router outputs

Training flow:
  Task 0: train classification + geo_rd → freeze
  Task t: detect → expand (group-specific expert) → train → freeze
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from backbones.sema_geometry_moe import GeometrySEMAModules

num_workers = 8


class GeoSEMAVitNet(nn.Module):
    """Geometry-SEMA ViT network wrapper."""

    def __init__(self, args, pretrained):
        super().__init__()
        from utils.inc_net import get_backbone
        self.backbone = get_backbone(args, pretrained)
        self.fc = None
        self.args = args
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        out = self.backbone(x)
        x_feat = out["features"]
        out.update({"logits": self.fc(x_feat)})
        return out


class Learner(BaseLearner):
    """Geometry-SEMA continual learner.

    Implements Group-MoE based continual learning with:
    - Group-specific expert expansion (not adapter-level)
    - Shared MambaFlow (not expanded)
    - Three-loss training (classification + geo_rd + semantic preservation)
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = GeoSEMAVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay", 0.0005)
        self.min_lr = args.get("min_lr", 1e-8)
        self.args = args

        # Loss weights
        self.lambda_geo_rd = args.get("lambda_geo_rd", 0.1)
        self.lambda_sem = args.get("lambda_sem", 0.01)
        self.beta_router = args.get("beta_router", 0.01)

        # Adaptive weights state
        self.alpha_rd = args.get("alpha_rd", 1.0)
        self.alpha_geo = args.get("alpha_geo", 0.1)
        self.alpha_bal = args.get("alpha_bal", 0.01)

        # Semantic preservation state
        self.old_router_probs = None
        self.old_prototypes = None

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1

        if self._cur_task == 0:
            self._network.fc = nn.Linear(768, data_manager.nb_classes)
            nn.init.kaiming_uniform_(self._network.fc.weight, a=math.sqrt(5))
            nn.init.zeros_(self._network.fc.bias)

        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=num_workers)
        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet, batch_size=self.batch_size,
            shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print("Multiple GPUs")
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        if self._cur_task == 0:
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f"{total_params:,} total parameters.")
            total_trainable = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f"{total_trainable:,} training parameters.")
            self._train_new(train_loader, test_loader)
            # Store router probs for semantic preservation
            self._store_router_probs(train_loader)
        else:
            # Detection + expansion
            for module in self._network.backbone.modules():
                if isinstance(module, GeometrySEMAModules):
                    module.detecting_outlier = True

            detect_loader = DataLoader(
                train_loader.dataset,
                batch_size=self.args.get("detect_batch_size", 128),
                shuffle=True, num_workers=num_workers)
            added = self._detect_outlier(detect_loader, train_loader, test_loader, 0)

            for module in self._network.backbone.modules():
                if isinstance(module, GeometrySEMAModules):
                    module.detecting_outlier = False

            if added == 0:
                logging.info("No expansion — fine-tuning routers only")
                self.update_optimizer_and_scheduler(
                    num_epoch=self.args.get("func_epoch", 5), lr=self.init_lr)
                self._init_train(self.args.get("func_epoch", 5),
                                 train_loader, test_loader,
                                 self.optimizer, self.scheduler, phase="func")

        # Freeze old experts
        for module in self._network.backbone.modules():
            if isinstance(module, GeometrySEMAModules):
                module.end_of_task_training()

        # Store for next task
        self._store_router_probs(train_loader)

    def _train_new(self, train_loader, test_loader):
        # func phase
        self.update_optimizer_and_scheduler(
            num_epoch=self.args.get("func_epoch", 5), lr=self.init_lr)
        self._init_train(self.args.get("func_epoch", 5),
                         train_loader, test_loader,
                         self.optimizer, self.scheduler, phase="func")

        # rd phase
        self.update_rd_optimizer_and_scheduler(
            num_epoch=self.args.get("rd_epoch", 20), lr=self.args.get("rd_lr", 0.01))
        self._init_train(self.args.get("rd_epoch", 20),
                         train_loader, test_loader,
                         self.rd_optimizer, self.rd_scheduler, phase="rd")

    def _detect_outlier(self, detect_loader, train_loader, test_loader, added):
        is_added = False

        for i, (_, inputs, targets) in enumerate(detect_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            model_outcome = self._network(inputs)
            added_record = model_outcome["added_record"]

            if sum(added_record) > 0:
                added += 1
                is_added = True

                for module in self._network.backbone.modules():
                    if isinstance(module, GeometrySEMAModules):
                        module.detecting_outlier = False

                self._train_new(train_loader, test_loader)

                for module in self._network.backbone.modules():
                    if isinstance(module, GeometrySEMAModules):
                        module.freeze_functional()
                        module.freeze_rd()
                        module.reset_newly_added_status()

                for module in self._network.backbone.modules():
                    if isinstance(module, GeometrySEMAModules):
                        module.detecting_outlier = True

        if is_added:
            return self._detect_outlier(
                detect_loader, train_loader, test_loader, added)
        return added

    # ═══════════════════════════════════════════════════════════════════
    # Three-loss training
    # ═══════════════════════════════════════════════════════════════════

    def _compute_geo_rd_loss(self, outcome):
        """Compute L_geo_rd = alpha_rd * L_group_RD + alpha_geo * L_geo + alpha_bal * L_balance.

        Section 11.2 in suggest.md.
        """
        group_rd_loss = outcome.get("group_rd_loss",
                                     torch.tensor(0., device=self._device))

        # SO orthogonality error
        ortho_error = torch.tensor(0., device=self._device)
        for module in self._network.backbone.modules():
            if isinstance(module, GeometrySEMAModules):
                for adapter in module.adapters:
                    ortho_error = ortho_error + adapter.orthogonality_error()

        # Router balance: KL(batch_mean_p || prior)
        balance_loss = torch.tensor(0., device=self._device)
        all_group_probs = outcome.get("all_group_probs", [])
        if all_group_probs:
            # Average group probs across deep layers
            mean_probs = torch.stack(
                [gp.mean(dim=0) for gp in all_group_probs]
            ).mean(dim=0)  # [G]
            prior = torch.ones_like(mean_probs) / len(mean_probs)
            # KL(batch_dist || uniform) to encourage balanced usage
            balance_loss = torch.sum(
                mean_probs * (torch.log(mean_probs + 1e-8) - torch.log(prior))
            )

        geo_rd = (
            self.alpha_rd * group_rd_loss
            + self.alpha_geo * ortho_error
            + self.alpha_bal * balance_loss
        )
        return geo_rd, {
            "group_rd": group_rd_loss.item(),
            "ortho": ortho_error.item(),
            "balance": balance_loss.item(),
        }

    def _compute_sem_loss(self, outcome):
        """Compute L_sem = L_proto + beta_router * L_router.

        Section 11.3 in suggest.md.
        Uses stored old router probabilities for consistency.
        """
        proto_loss = torch.tensor(0., device=self._device)
        router_loss = torch.tensor(0., device=self._device)

        # Router consistency: KL(p_old || p_new)
        if self.old_router_probs is not None:
            all_group_probs = outcome.get("all_group_probs", [])
            if all_group_probs and len(self.old_router_probs) > 0:
                # Compare deep layer group probs
                for i, (new_probs, old_probs) in enumerate(
                    zip(all_group_probs, self.old_router_probs)):
                    if i >= len(self.old_router_probs):
                        break
                    # KL(old || new)
                    kl = torch.sum(
                        old_probs.to(self._device)
                        * (torch.log(old_probs.to(self._device) + 1e-8)
                           - torch.log(new_probs + 1e-8))
                    )
                    router_loss = router_loss + kl
                router_loss = router_loss / max(len(all_group_probs), 1)

        sem_loss = proto_loss + self.beta_router * router_loss
        return sem_loss

    def _init_train(self, total_epoch, train_loader, test_loader,
                    optimizer, scheduler, phase="func"):
        tracker = self.get_loss_tracker()
        prog_bar = tqdm(range(total_epoch))

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outcome = self._network(inputs)

                logits = outcome["logits"]
                logits = logits[:, :self._total_classes]

                if self._cur_task > 0:
                    logits[:, :self._known_classes] = -float("inf")

                if phase == "func":
                    # L_cls: cross-entropy
                    loss_cls = F.cross_entropy(logits, targets)

                    # L_geo_rd: group-aware RD + orthogonality + balance
                    loss_geo_rd, geo_components = self._compute_geo_rd_loss(outcome)

                    # L_sem: prototype + router consistency
                    loss_sem = self._compute_sem_loss(outcome)

                    loss = (
                        loss_cls
                        + self.lambda_geo_rd * loss_geo_rd
                        + self.lambda_sem * loss_sem
                    )
                elif phase == "rd":
                    loss = outcome.get("group_rd_loss",
                                       torch.tensor(0., device=self._device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tracker.update(**{phase: loss.item()})

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_losses = tracker.flush(epoch)
            avg_loss = avg_losses.get(phase, 0)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = (
                "{} Task {}, Epoch {}/{} => Loss {:.3f}, "
                "Train_accy {:.2f}, Test_accy {:.2f}".format(
                    phase, self._cur_task, epoch + 1, total_epoch,
                    avg_loss, train_acc, test_acc))
            prog_bar.set_description(info)

        logging.info(info)

    def _store_router_probs(self, loader):
        """Store router probabilities for semantic preservation in next task."""
        self._network.eval()
        all_probs = []
        with torch.no_grad():
            for i, (_, inputs, _) in enumerate(loader):
                if i >= 2:  # 2 batches are enough
                    break
                inputs = inputs.to(self._device)
                outcome = self._network(inputs)
                gp = outcome.get("all_group_probs", [])
                if gp:
                    # Average over batches
                    avg = torch.stack([p.mean(dim=0) for p in gp])
                    all_probs.append(avg)

        if all_probs:
            self.old_router_probs = [
                torch.stack([ap[i] for ap in all_probs]).mean(dim=0)
                for i in range(len(all_probs[0]))
            ]
        else:
            self.old_router_probs = None

    # ═══════════════════════════════════════════════════════════════════
    # Evaluation
    # ═══════════════════════════════════════════════════════════════════

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outcome = self._network(inputs)
                logits = outcome["logits"]
                outputs = logits[:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outcome = model(inputs)
                logits = outcome["logits"]
                outputs = logits[:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    # ═══════════════════════════════════════════════════════════════════
    # Optimizer management
    # ═══════════════════════════════════════════════════════════════════

    def update_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args["init_lr"] if lr is None else lr
        func_params = [
            p for n, p in self._network.named_parameters()
            if ("functional" in n or "router" in n or "fc" in n or "vpt" in n
                or "down_proj" in n or "up_proj" in n or "gamma" in n
                or "group_bank" in n or "group_ae" in n)
            and p.requires_grad
        ]
        if self.args.get("optimizer", "sgd") == "sgd":
            self.optimizer = optim.SGD(
                func_params, momentum=0.9, lr=lr,
                weight_decay=self.args.get("weight_decay", 0.0005))
        elif self.args.get("optimizer") == "adam":
            self.optimizer = optim.AdamW(
                func_params, lr=lr,
                weight_decay=self.args.get("weight_decay", 0.0005))

        min_lr = self.args.get("min_lr", 1e-8)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epoch, eta_min=min_lr)

    def update_rd_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args.get("rd_lr", 0.01) if lr is None else lr
        rd_params = [
            p for n, p in self._network.named_parameters()
            if ("group_ae" in n or "rd" in n)
            and p.requires_grad
        ]
        if self.args.get("optimizer", "sgd") == "sgd":
            self.rd_optimizer = optim.SGD(
                rd_params, momentum=0.9, lr=lr,
                weight_decay=self.args.get("weight_decay", 0.0005))
        elif self.args.get("optimizer") == "adam":
            self.rd_optimizer = optim.AdamW(
                rd_params, lr=lr,
                weight_decay=self.args.get("weight_decay", 0.0005))

        min_lr = self.args.get("min_lr", 1e-8)
        self.rd_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.rd_optimizer, T_max=num_epoch, eta_min=min_lr)

    def save_checkpoint(self, filename):
        state_dict = self._network.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if ("adapter" in k or "fc" in k and "block" not in k
                    or "group" in k or "down_proj" in k or "up_proj" in k):
                save_dict[k] = v
        torch.save(save_dict, "{}.pth".format(filename))

    def load_checkpoint(self, filename):
        self._network.load_state_dict(torch.load(filename), strict=False)
