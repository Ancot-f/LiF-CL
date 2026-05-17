"""
Flat MoE Learner — 扁平混合专家持续学习器
===========================================

训练流程 (继承 SEMA 的三阶段):
  Task 0: func phase (L_cls) → rd phase (L_rd)
  Task t: detect → expand → train func → train rd → freeze

损失:
  L_total = L_cls + lambda_rd * L_rd + lambda_geo * L_geo + lambda_sem * L_sem

组件:
  - FlatMoEVitNet: 网络包装器
  - Learner: 训练器
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
from backbones.flat_moe import ExpertMoEModules

num_workers = 8


class FlatMoEVitNet(nn.Module):
    """Flat MoE ViT 网络包装器。"""

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

    def forward(self, x):
        out = self.backbone(x)
        x_feat = out["features"]
        out.update({"logits": self.fc(x_feat)})
        return out


class Learner(BaseLearner):
    """Flat MoE 持续学习器。

    与 Geo-SEMA 的区别:
      - 无 Group-MoE (无群结构、无层次路由)
      - 无 MambaFlow (无 selective_scan 循环)
      - 专家 = 瓶颈 MLP
      - 扁平路由: 1 层 softmax
      - 逐专家 RD
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = FlatMoEVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay", 0.0005)
        self.min_lr = args.get("min_lr", 1e-8)
        self.args = args

        self.lambda_rd = args.get("lambda_rd", 0.1)
        self.lambda_geo = args.get("lambda_geo", 0.01)
        self.lambda_sem = args.get("lambda_sem", 0.01)
        self.beta_router = args.get("beta_router", 0.01)

        self.old_router_probs = None

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1

        if self._cur_task == 0:
            self._network.fc = nn.Linear(768, data_manager.nb_classes)
            nn.init.kaiming_uniform_(self._network.fc.weight, a=math.sqrt(5))
            nn.init.zeros_(self._network.fc.bias)

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=num_workers)

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
            total_trainable = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f"{total_trainable:,} training parameters.")
            self._train_new(train_loader, test_loader)
            self._store_router_probs(train_loader)
        else:
            for module in self._network.backbone.modules():
                if isinstance(module, ExpertMoEModules):
                    module.detecting_outlier = True

            detect_loader = DataLoader(train_loader.dataset,
                                       batch_size=self.args.get("detect_batch_size", 128),
                                       shuffle=True, num_workers=num_workers)
            added = self._detect_outlier(detect_loader, train_loader, test_loader, 0)

            for module in self._network.backbone.modules():
                if isinstance(module, ExpertMoEModules):
                    module.detecting_outlier = False

            if added == 0:
                logging.info("No expansion — fine-tuning routers + classifier")
                self.update_optimizer_and_scheduler(
                    num_epoch=self.args.get("func_epoch", 5), lr=self.init_lr)
                self._init_train(self.args.get("func_epoch", 5),
                                 train_loader, test_loader,
                                 self.optimizer, self.scheduler, phase="func")

        for module in self._network.backbone.modules():
            if isinstance(module, ExpertMoEModules):
                module.end_of_task_training()

        self._store_router_probs(train_loader)

    def _train_new(self, train_loader, test_loader):
        # func phase
        self.update_optimizer_and_scheduler(
            num_epoch=self.args.get("func_epoch", 5), lr=self.init_lr)
        self._init_train(self.args.get("func_epoch", 5),
                         train_loader, test_loader,
                         self.optimizer, self.scheduler, phase="func")

        # rd phase
        for module in self._network.backbone.modules():
            if isinstance(module, ExpertMoEModules):
                module._training_rd = True
        self.update_rd_optimizer_and_scheduler(
            num_epoch=self.args.get("rd_epoch", 20), lr=self.args.get("rd_lr", 0.01))
        self._init_train(self.args.get("rd_epoch", 20),
                         train_loader, test_loader,
                         self.rd_optimizer, self.rd_scheduler, phase="rd")
        for module in self._network.backbone.modules():
            if isinstance(module, ExpertMoEModules):
                module._training_rd = False

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
                    if isinstance(module, ExpertMoEModules):
                        module.detecting_outlier = False

                self._train_new(train_loader, test_loader)

                for module in self._network.backbone.modules():
                    if isinstance(module, ExpertMoEModules):
                        module.freeze_functional()
                        module.freeze_rd()
                        module.reset_newly_added_status()

                for module in self._network.backbone.modules():
                    if isinstance(module, ExpertMoEModules):
                        module.detecting_outlier = True

        if is_added:
            return self._detect_outlier(detect_loader, train_loader, test_loader, added)
        return added

    # ═══════════════════════════════════════════════════════════════════
    # 损失计算
    # ═══════════════════════════════════════════════════════════════════

    def _compute_geo_loss(self):
        """计算 SO + LR 正则化损失。"""
        so_loss = torch.tensor(0., device=self._device)
        lr_loss = torch.tensor(0., device=self._device)
        for module in self._network.backbone.modules():
            if isinstance(module, ExpertMoEModules):
                for adapter in module.adapters:
                    so_loss = so_loss + adapter.orthogonality_error()
                    lr_loss = lr_loss + adapter.low_rank_error()
        return so_loss + lr_loss

    def _compute_sem_loss(self, outcome):
        """路由一致性损失: KL(p_old || p_new)。"""
        router_loss = torch.tensor(0., device=self._device)
        if self.old_router_probs is not None:
            # Flat MoE 只有一层 expert_weights, 直接比较
            # old_router_probs 是 list[Tensor], 取平均
            pass
        return router_loss

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
                    loss_rd = outcome.get("rd_loss", torch.tensor(0., device=self._device))
                    loss_geo = self._compute_geo_loss()
                    loss_sem = self._compute_sem_loss(outcome)
                    loss = (loss_cls + self.lambda_rd * loss_rd
                            + self.lambda_geo * loss_geo + self.lambda_sem * loss_sem)
                elif phase == "rd":
                    loss = outcome.get("rd_loss", torch.tensor(0., device=self._device))

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
            info = ("{} Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                phase, self._cur_task, epoch + 1, total_epoch, avg_loss, train_acc, test_acc))
            prog_bar.set_description(info)

        logging.info(info)

    def _store_router_probs(self, loader):
        """存储当前专家权重分布 (用于语义保持)。"""
        self._network.eval()
        with torch.no_grad():
            for i, (_, inputs, _) in enumerate(loader):
                if i >= 2:
                    break
                inputs = inputs.to(self._device)
                self._network(inputs)
        # Flat MoE: 路由分布从 expert_weights 获取
        self.old_router_probs = None  # 简化实现

    # ═══════════════════════════════════════════════════════════════════
    # 评估
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
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
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
    # 优化器管理
    # ═══════════════════════════════════════════════════════════════════

    def update_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args["init_lr"] if lr is None else lr
        func_params = [
            p for n, p in self._network.named_parameters()
            if (p.requires_grad and (
                "router" in n or "fc" in n or "vpt" in n
                or "expert" in n or "expert_bank" in n
                or "expert_ae" in n or "gamma" in n))
        ]
        if not func_params:
            logging.warning("No trainable params for func phase, using fc only")
            func_params = [p for n, p in self._network.named_parameters()
                          if p.requires_grad]
        opt_name = self.args.get("optimizer", "sgd")
        if opt_name == "sgd":
            self.optimizer = optim.SGD(func_params, momentum=0.9, lr=lr,
                                       weight_decay=self.args.get("weight_decay", 0.0005))
        elif opt_name == "adam":
            self.optimizer = optim.AdamW(func_params, lr=lr,
                                         weight_decay=self.args.get("weight_decay", 0.0005))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epoch, eta_min=self.args.get("min_lr", 1e-8))

    def update_rd_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args.get("rd_lr", 0.01) if lr is None else lr
        rd_params = [
            p for n, p in self._network.named_parameters()
            if ("expert_ae" in n or "rd" in n) and p.requires_grad
        ]
        if not rd_params:
            logging.warning("No trainable params for rd phase")
            rd_params = [p for n, p in self._network.named_parameters()
                        if "expert_ae" in n]
        opt_name = self.args.get("optimizer", "sgd")
        if opt_name == "sgd":
            self.rd_optimizer = optim.SGD(rd_params, momentum=0.9, lr=lr,
                                          weight_decay=self.args.get("weight_decay", 0.0005))
        elif opt_name == "adam":
            self.rd_optimizer = optim.AdamW(rd_params, lr=lr,
                                            weight_decay=self.args.get("weight_decay", 0.0005))
        self.rd_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.rd_optimizer, T_max=num_epoch, eta_min=self.args.get("min_lr", 1e-8))

    def save_checkpoint(self, filename):
        state_dict = self._network.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if ("adapter" in k or "fc" in k or "expert" in k):
                save_dict[k] = v
        torch.save(save_dict, "{}.pth".format(filename))

    def load_checkpoint(self, filename):
        self._network.load_state_dict(torch.load(filename), strict=False)
