"""
李群 SEMA 学习器 (Lie Group Fiber Bundle SEMA Learner)
======================================================

基于 SEMA 的标准训练流程, 增加 Stiefel 流形约束:
  - Adapter.down_proj 约束在 St(768, 16) 上
  - 每次 optimizer.step() 后将可训练 Adapter 的 down_proj 投影回 Stiefel
  - 扩展检测: 测地线距离替代 Z-score

与标准 SEMA (models/sema.py) 的关键区别:
  1. update_optimizer_and_scheduler → 不变 (采相同的参数筛选)
  2. _init_train → 在 optimizer.step() 后调用 _project_trainable()
  3. _detect_outlier → 扩展检测改为测地线距离判断 (check_geodesic_expansion)
  4. _train_new → 新增测地线距离日志
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

from utils.inc_net import BaseNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from backbones.lie_sema_block import LieSEMAModules

num_workers = 8


class LieSEMAVitNet(BaseNet):
    """Lie-SEMA 网络包装器 — 与 SEMAVitNet 相同但使用 Lie-SEMA backbone。"""

    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.fc = None
        self.args = args

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        out = self.backbone(x)
        x_feat = out["features"]
        out.update({"logits": self.fc(x_feat)})
        return out


class Learner(BaseLearner):
    """Lie-SEMA 持续学习器。

    训练流程 (每个任务):
      1. Task 0: func 阶段 + rd 阶段 → Stiefel 投影 → 冻结
      2. Task > 0:
         a. 检测模式: 尝试训练新 Adapter, 用测地线距离判断是否扩展
         b. 若扩展 → _train_new → Stiefel 投影 → freeze
         c. 若未扩展 → 只微调 Router (不做 Stiefel 投影, Adapter 不变)
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = LieSEMAVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay", 0.0005)
        self.min_lr = args.get("min_lr", 1e-8)
        self.args = args

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
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        if self._cur_task == 0:
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
            self._train_new(train_loader, test_loader)
        else:
            # 扩展检测: 先训练新 Adapter, 再用测地线距离判断是否存在有效扩展
            for module in self._network.backbone.modules():
                if isinstance(module, LieSEMAModules):
                    module.detecting_outlier = True

            detect_loader = DataLoader(
                train_loader.dataset,
                batch_size=self.args.get("detect_batch_size", 128),
                shuffle=True, num_workers=num_workers)

            added = self._detect_outlier(detect_loader, train_loader, test_loader, 0)

            for module in self._network.backbone.modules():
                if isinstance(module, LieSEMAModules):
                    module.detecting_outlier = False

            if added == 0:
                logging.info("No expansion triggered — fine-tuning Router only")
                self.update_optimizer_and_scheduler(
                    num_epoch=self.args.get("func_epoch", 20), lr=self.init_lr)
                self._init_train(self.args.get("func_epoch", 20),
                                 train_loader, test_loader,
                                 self.optimizer, self.scheduler, phase="func")

        # 任务结束后: 先将所有 Adapter 投影到 Stiefel, 再冻结
        self._network.backbone.project_all_adapters_()
        for module in self._network.backbone.modules():
            if isinstance(module, LieSEMAModules):
                module.end_of_task_training()

    def _train_new(self, train_loader, test_loader):
        self.update_optimizer_and_scheduler(
            num_epoch=self.args.get("func_epoch", 20), lr=self.init_lr)
        self._init_train(self.args.get("func_epoch", 20),
                         train_loader, test_loader,
                         self.optimizer, self.scheduler, phase="func")
        self.update_rd_optimizer_and_scheduler(
            num_epoch=self.args.get("rd_epoch", 20), lr=self.args.get("rd_lr", 0.01))
        self._init_train(self.args.get("rd_epoch", 20),
                         train_loader, test_loader,
                         self.rd_optimizer, self.rd_scheduler, phase="rd")

    def _detect_outlier(self, detect_loader, train_loader, test_loader, added):
        """扩展检测 — 自由训练, 然后投影到 Stiefel 用测地线距离判断。

        与标准 SEMA 的区别:
          SEMA:  训练时计算 Z-score, batch 级别检测
          Lie-SEMA: 自由训练完成后, 将 down_proj 投影到 Stiefel 流形,
                    用测地线距离判断是否与旧 Adapter "太远"。

        流程:
          1. 在检测层添加临时 Adapter, 自由训练 (func + rd)
          2. project_all_adapters_() 将所有可训练 down_proj 投影到 Stiefel
          3. check_geodesic_expansion() 计算 min 测地线距离
          4. 距离 > threshold → 确认扩展; 否则回滚
        """
        is_added = False

        for i, (_, inputs, targets) in enumerate(detect_loader):
            if i > 0:
                break  # 只在第一个 batch 做一次检测

            inputs, targets = inputs.to(self._device), targets.to(self._device)
            backbone = self._network.backbone

            # 获取所有 LieSEMAModules
            sema_modules = [m for m in backbone.modules()
                            if isinstance(m, LieSEMAModules)]

            # 检查每个检测层（自顶向下: 9→10→11）
            any_expand = False
            for sm in sema_modules:
                if not (sm.layer_id >= self.args.get("adapt_start_layer", 9)
                        and sm.layer_id <= self.args.get("adapt_end_layer", 11)):
                    continue

                # 本层已在本任务中扩展过，跳过
                if sm.added_for_task:
                    continue

                sm.detecting_outlier = False
                sm.add_adapter()  # 内部设置 added_for_task = True
                self._train_new(train_loader, test_loader)

                # ── 投影到 Stiefel 后做几何判断 ──
                backbone.project_all_adapters_()

                should_expand, min_dist = sm.check_geodesic_expansion()
                logging.info(
                    f"Block {sm.layer_id}: geo_min_dist={min_dist:.4f}, "
                    f"threshold={sm.geo_threshold}, expand={should_expand}")

                if should_expand:
                    any_expand = True
                    added += 1
                    is_added = True
                    sm.freeze_functional()
                    sm.freeze_rd()
                    sm.reset_newly_added_status()
                    # 不重置 added_for_task — 保持 True 防止递归中再次扩展本层
                    break  # 每次最多扩展一层
                else:
                    # 回滚: 删除临时 Adapter, 重置标志
                    logging.info(f"Block {sm.layer_id}: rollback — geodesic distance too small")
                    sm.adapters.pop(-1)
                    sm.new_router = None
                    sm.added_for_task = False  # 回滚后允许后续该层被重新检测
                    break

            if any_expand:
                return self._detect_outlier(detect_loader, train_loader, test_loader, added)
            break

        return added

    def _init_train(self, total_epoch, train_loader, test_loader,
                    optimizer, scheduler, phase="func"):
        """单阶段训练循环 — Riemannian SGD，每个 step 后将 down_proj 投影回 Stiefel。"""
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
                    loss = F.cross_entropy(logits, targets)
                elif phase == "rd":
                    loss = outcome["rd_loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # func 阶段: 每个 step 后将可训练 Adapter 的 down_proj 投影回 Stiefel
                if phase == "func":
                    self._network.backbone.project_all_adapters_()

                tracker.update(**{phase: loss.item()})

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_losses = tracker.flush(epoch)
            avg_loss = avg_losses.get(phase, 0)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = ("{} Task {}, Epoch {}/{} => Loss {:.3f}, "
                    "Train_accy {:.2f}, Test_accy {:.2f}".format(
                        phase, self._cur_task, epoch + 1, total_epoch,
                        avg_loss, train_acc, test_acc))
            prog_bar.set_description(info)

        logging.info(info)

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

    def update_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args["init_lr"] if lr is None else lr
        func_params = [
            p for n, p in self._network.named_parameters()
            if ("functional" in n or "router" in n or "fc" in n or "vpt" in n)
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
            if "rd" in n and p.requires_grad
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
            self.rd_optimizer, T_max=num_epoch, eta_min=min_lr) if self.rd_optimizer else None

    def save_checkpoint(self, filename):
        state_dict = self._network.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if "adapter" in k or ("fc" in k and "block" not in k):
                save_dict[k] = v
        torch.save(save_dict, "{}.pth".format(filename))

    def load_checkpoint(self, filename):
        self._network.load_state_dict(torch.load(filename), strict=False)
