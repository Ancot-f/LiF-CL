"""
Sparse Geometry-Aware MoE Learner
==================================

Training pipeline for Sparse Group-MoE:
  - Task 0: two-phase training (func + rd)
  - Task t: detect + expand + train
  - Three-level loss: L_cls + lambda_rd * L_group_rd + lambda_sem * L_sem

Key differences from geo_sema.py:
  - Uses SparseGroupMoEModules (top-k group selection)
  - Tracks selected_mask for monitoring sparse routing behavior
  - Expansion is per-group expert level within selected groups
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
from backbones.sparse_geo_moe import SparseGroupMoEModules

num_workers = 8


class SparseGeoMoEVitNet(nn.Module):
    """Sparse Group-MoE ViT network wrapper.

    Encapsulates backbone + fc head, provides unified forward interface.
    """

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
    """Sparse Geometry-Aware MoE continual learner.

    Implements:
      - Sparse top-k group selection with z-score feedback
      - Intra-group expert expansion (not adapter-level)
      - Shared MambaFlow (not expanded)
      - Three-level loss (classification + group-RD + semantic)

    Training flow:
      Task 0: func phase → rd phase
      Task t: detect → (expand + train) or (fine-tune routers)
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = SparseGeoMoEVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay", 0.0005)
        self.min_lr = args.get("min_lr", 1e-8)
        self.args = args

        # Loss weights
        self.lambda_geo_rd = args.get("lambda_geo_rd", 0.1)
        self.lambda_sem = args.get("lambda_sem", 0.01)
        self.beta_router = args.get("beta_router", 0.01)

        # L_geo_rd internal adaptive weights
        self.alpha_rd = args.get("alpha_rd", 1.0)
        self.alpha_geo = args.get("alpha_geo", 0.1)
        self.alpha_bal = args.get("alpha_bal", 0.01)

        # Adaptive weight ranges
        self.alpha_rd_min = args.get("alpha_rd_min", 0.1)
        self.alpha_rd_max = args.get("alpha_rd_max", 10.0)
        self.alpha_geo_min = args.get("alpha_geo_min", 0.01)
        self.alpha_geo_max = args.get("alpha_geo_max", 1.0)
        self.alpha_bal_min = args.get("alpha_bal_min", 0.001)
        self.alpha_bal_max = args.get("alpha_bal_max", 0.1)

        # Adaptive weight modulation
        self.gamma_z = args.get("gamma_z", 0.5)
        self.gamma_o = args.get("gamma_o", 1.0)
        self.gamma_u = args.get("gamma_u", 1.0)

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
            "══════════════ Task {}: classes {}-{} ══════════════".format(
                self._cur_task, self._known_classes, self._total_classes))

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
            self._store_router_probs(train_loader)
        else:
            # Enable outlier detection
            for module in self._network.backbone.modules():
                if isinstance(module, SparseGroupMoEModules):
                    module.detecting_outlier = True

            detect_loader = DataLoader(
                train_loader.dataset,
                batch_size=self.args.get("detect_batch_size", 128),
                shuffle=True, num_workers=num_workers)
            added = self._detect_outlier(detect_loader, train_loader, test_loader, 0)

            for module in self._network.backbone.modules():
                if isinstance(module, SparseGroupMoEModules):
                    module.detecting_outlier = False

            # Detection summary: per-layer z-score / entropy / best group
            detect_summary = []
            for module in self._network.backbone.modules():
                if isinstance(module, SparseGroupMoEModules):
                    za = module._z_score_accum
                    best_idx = za.argmax().item()
                    gn = module.adapters[-1].idx_to_group_name[best_idx]
                    detect_summary.append(
                        f"L{module.layer_id}(z={za[best_idx]:.2f} "
                        f"H={module._entropy_ema:.3f}→{gn})"
                    )
            logging.info(
                f"Task {self._cur_task} detect: {' | '.join(detect_summary)} "
                f"→ expanded={added}")

            if added == 0:
                logging.info("No expansion — fine-tuning routers only")
                self.update_optimizer_and_scheduler(
                    num_epoch=self.args.get("func_epoch", 5), lr=self.init_lr,
                    expanded=False)
                self._init_train(self.args.get("func_epoch", 5),
                                 train_loader, test_loader,
                                 self.optimizer, self.scheduler, phase="func")

        for module in self._network.backbone.modules():
            if isinstance(module, SparseGroupMoEModules):
                module.end_of_task_training()

        # Log per-layer expert counts after each task
        layer_info = []
        for m in self._network.backbone.modules():
            if isinstance(m, SparseGroupMoEModules):
                ec = m.expansion_count
                layer_info.append(
                    f"L{m.layer_id}:[ID=1 SO={1+ec['SO']} "
                    f"LR={1+ec['LR']} Aff={1+ec['Affine']}]"
                )
        logging.info(f"Task {self._cur_task} experts: " + " ".join(layer_info))

        self._store_router_probs(train_loader)

    def _train_new(self, train_loader, test_loader):
        """Two-phase training: func → rd. Called for Task 0 or after expansion."""
        # Phase 1: functional training (classification)
        # Task 0: train everything to establish baseline
        # Post-expansion: only train Router + new expert
        is_task0 = (self._cur_task == 0)
        self.update_optimizer_and_scheduler(
            num_epoch=self.args.get("func_epoch", 5), lr=self.init_lr,
            expanded=is_task0, is_task0=is_task0)
        self._init_train(self.args.get("func_epoch", 5),
                         train_loader, test_loader,
                         self.optimizer, self.scheduler, phase="func")

        # Phase 2: RD training (representation descriptor)
        for module in self._network.backbone.modules():
            if isinstance(module, SparseGroupMoEModules):
                module._training_rd = True
        self.update_rd_optimizer_and_scheduler(
            num_epoch=self.args.get("rd_epoch", 20),
            lr=self.args.get("rd_lr", 0.01))
        self._init_train(self.args.get("rd_epoch", 20),
                         train_loader, test_loader,
                         self.rd_optimizer, self.rd_scheduler, phase="rd")
        for module in self._network.backbone.modules():
            if isinstance(module, SparseGroupMoEModules):
                module._training_rd = False

    def _detect_outlier(self, detect_loader, train_loader, test_loader, added):
        """Outlier detection + recursive expansion."""
        is_added = False

        for i, (_, inputs, targets) in enumerate(detect_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            model_outcome = self._network(inputs)
            added_record = model_outcome["added_record"]

            if sum(added_record) > 0:
                added += 1
                is_added = True

                for module in self._network.backbone.modules():
                    if isinstance(module, SparseGroupMoEModules):
                        module.detecting_outlier = False

                self._train_new(train_loader, test_loader)

                for module in self._network.backbone.modules():
                    if isinstance(module, SparseGroupMoEModules):
                        module.freeze_functional()
                        module.freeze_rd()
                        module.reset_newly_added_status()

                for module in self._network.backbone.modules():
                    if isinstance(module, SparseGroupMoEModules):
                        module.detecting_outlier = True

        if is_added:
            return self._detect_outlier(
                detect_loader, train_loader, test_loader, added)
        return added

    # ═══════════════════════════════════════════════════════════════════
    # Loss computation
    # ═══════════════════════════════════════════════════════════════════

    def _compute_adaptive_weights(self, outcome):
        all_z_scores = outcome.get("all_z_scores", [])
        z_norm = 0.0
        if all_z_scores:
            z_stack = torch.cat([zs.mean(dim=0) for zs in all_z_scores])
            z_norm = z_stack.norm().item()

        orth_error = 0.0
        for module in self._network.backbone.modules():
            if isinstance(module, SparseGroupMoEModules):
                for adapter in module.adapters:
                    orth_error += adapter.orthogonality_error().item()

        usage_imbalance = 0.0
        all_group_probs = outcome.get("all_group_probs", [])
        if all_group_probs:
            mean_probs = torch.stack(
                [gp.mean(dim=0) for gp in all_group_probs]
            ).mean(dim=0)
            G = len(mean_probs)
            prior = torch.ones(G, device=mean_probs.device) / G
            usage_imbalance = torch.sum(
                mean_probs.detach()
                * (torch.log(mean_probs.detach() + 1e-8) - torch.log(prior))
            ).item()

        alpha_rd = np.clip(
            1.0 + self.gamma_z * z_norm,
            self.alpha_rd_min, self.alpha_rd_max)
        alpha_geo = np.clip(
            self.alpha_geo * (1.0 + self.gamma_o * orth_error),
            self.alpha_geo_min, self.alpha_geo_max)
        alpha_bal = np.clip(
            self.alpha_bal * (1.0 + self.gamma_u * usage_imbalance),
            self.alpha_bal_min, self.alpha_bal_max)

        return alpha_rd, alpha_geo, alpha_bal

    def _compute_geo_rd_loss(self, outcome):
        alpha_rd, alpha_geo, alpha_bal = self._compute_adaptive_weights(outcome)

        group_rd_loss = outcome.get("group_rd_loss",
                                    torch.tensor(0., device=self._device))

        ortho_error = torch.tensor(0., device=self._device)
        for module in self._network.backbone.modules():
            if isinstance(module, SparseGroupMoEModules):
                for adapter in module.adapters:
                    ortho_error = ortho_error + adapter.orthogonality_error()

        # Deviation penalty: prevent capacity-driven group drift (esp. Affine)
        beta_dev = self.args.get("beta_dev", 0.01)
        dev_penalty = torch.tensor(0., device=self._device)
        for module in self._network.backbone.modules():
            if isinstance(module, SparseGroupMoEModules):
                for adapter in module.adapters:
                    dev_penalty = dev_penalty + adapter.group_bank.deviation_penalty()

        balance_loss = torch.tensor(0., device=self._device)
        all_group_probs = outcome.get("all_group_probs", [])
        if all_group_probs:
            mean_probs = torch.stack(
                [gp.mean(dim=0) for gp in all_group_probs]
            ).mean(dim=0)
            G = len(mean_probs)
            prior = torch.ones(G, device=mean_probs.device) / G
            balance_loss = torch.sum(
                mean_probs * (torch.log(mean_probs + 1e-8) - torch.log(prior))
            )

        geo_rd = (
            alpha_rd * group_rd_loss
            + alpha_geo * ortho_error
            + alpha_bal * balance_loss
            + beta_dev * dev_penalty
        )

        return geo_rd, {
            "group_rd": group_rd_loss.item(),
            "ortho": ortho_error.item(),
            "balance": balance_loss.item(),
            "alpha_rd": alpha_rd,
            "alpha_geo": alpha_geo,
            "alpha_bal": alpha_bal,
        }

    def _compute_sem_loss(self, outcome):
        proto_loss = torch.tensor(0., device=self._device)
        router_loss = torch.tensor(0., device=self._device)
        flow_loss = torch.tensor(0., device=self._device)

        if self.old_router_probs is not None:
            all_group_probs = outcome.get("all_group_probs", [])
            if all_group_probs and len(self.old_router_probs) > 0:
                for i, (new_probs, old_probs) in enumerate(
                    zip(all_group_probs, self.old_router_probs)):
                    if i >= len(self.old_router_probs):
                        break
                    kl = torch.sum(
                        old_probs.to(self._device)
                        * (torch.log(old_probs.to(self._device) + 1e-8)
                           - torch.log(new_probs + 1e-8))
                    )
                    router_loss = router_loss + kl
                router_loss = router_loss / max(len(all_group_probs), 1)

        sem_loss = proto_loss + self.beta_router * router_loss + flow_loss
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
                    loss_cls = F.cross_entropy(logits, targets)
                    loss_geo_rd, geo_components = self._compute_geo_rd_loss(outcome)
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
        self._network.eval()
        all_probs = []
        with torch.no_grad():
            for i, (_, inputs, _) in enumerate(loader):
                if i >= 2:
                    break
                inputs = inputs.to(self._device)
                outcome = self._network(inputs)
                gp = outcome.get("all_group_probs", [])
                if gp:
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

    def update_optimizer_and_scheduler(self, num_epoch=20, lr=None,
                                        expanded=False, is_task0=False):
        lr = self.args["init_lr"] if lr is None else lr
        if is_task0:
            # Task 0: train full adapter to establish baseline
            train_keys = ["down_proj", "up_proj", "gamma", "router",
                          "fc", "vpt", "group_ae", "mamba", "group_bank"]
        elif expanded:
            train_keys = ["router", "fc", "gamma", "group_bank",
                          "down_proj", "up_proj"]
        else:
            train_keys = ["router", "fc", "gamma"]

        func_params = [
            p for n, p in self._network.named_parameters()
            if any(k in n for k in train_keys) and p.requires_grad
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
            if ("group_ae" in n or "rd" in n)  # AE only, NOT group_bank
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
                    or "group" in k or "down_proj" in k or "up_proj" in k
                    or "mamba" in k):
                save_dict[k] = v
        torch.save(save_dict, "{}.pth".format(filename))

    def load_checkpoint(self, filename):
        self._network.load_state_dict(torch.load(filename), strict=False)
