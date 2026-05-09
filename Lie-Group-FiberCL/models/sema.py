"""
SEMA 学习器
==========

实现 Self-Expansion of Pre-trained Models with Mixture of Adapters (SEMA)
的完整训练流程。

核心算法:
  1. 第一个任务: 初始化 ViT + 每层各一个 Adapter
  2. 训练流程:
     a. func 阶段: 冻结主干 + 旧 Adapter，只训练:
        - 新 Adapter 的 functional 模块
        - 新 Router 列
        - 分类器头 (fc)
        损失: 交叉熵
     b. rd 阶段: 训练新 Adapter 的表征描述器 (AE)
        损失: 重建损失 MSE(x, AE(x))
  3. 自扩展检测（后续任务）:
     a. 开启检测模式 (detecting_outlier = True)
     b. 用所有旧 RD 评估当前数据
     c. 如果所有旧 RD 的 Z-score 都超过阈值 → 触发扩展
     d. 自顶向下扫描，第一个触发的层添加新 Adapter
  4. 冻结:
     a. 训练完后 freeze_functional() → 旧 Adapter 权重锁定
     b. freeze_rd() → 旧 RD 权重锁定 + 统计停止更新
     c. fix_router() → 新 Router 列合并到主 Router

关键设计:
  - 每次最多添加 1 个 Adapter → 子线性参数增长
  - 旧 Adapter 完全冻结 → 零遗忘（不需要 replay）
  - 软路由组合 → 避免硬选择的错误传播
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

from utils.inc_net import SEMAVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from backbones.sema_block import SEMAModules

num_workers = 8  # DataLoader 线程数


class Learner(BaseLearner):
    """SEMA 持续学习器。

    继承 BaseLearner，实现 SEMA 的自扩展训练逻辑。

    训练流程（每个任务）:
    1. 第一个任务: _train_new(func 阶段 + rd 阶段)
    2. 后续任务:
       a. _detect_outlier() → 检测是否需要扩展
       b. 如果扩展了 → _train_new (训练新增的 Adapter)
       c. 如果没扩展 → _init_train (只用路由器组合已有 Adapter)
    """

    def __init__(self, args):
        super().__init__(args)
        self._network = SEMAVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = (
            args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        )
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args

    def after_task(self):
        """任务完成回调: 更新已知类别数。"""
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        """一个增量任务的入口。

        负责:
        - 设置当前任务的数据加载器
        - 更新类别计数
        - 调度 _train() 进行实际训练
        """
        self._cur_task += 1

        # 第一个任务: 初始化分类器头
        if self._cur_task == 0:
            self._network.fc = nn.Linear(768, data_manager.nb_classes)
            nn.init.kaiming_uniform_(self._network.fc.weight, a=math.sqrt(5))
            nn.init.zeros_(self._network.fc.bias)

        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # 构建数据加载器
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="train",
        )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=num_workers,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test",
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=num_workers,
        )

        # 可选: protonet 加载器（用于 NME 评估的类原型计算）
        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="test",
        )
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet, batch_size=self.batch_size,
            shuffle=True, num_workers=num_workers,
        )

        # 多 GPU 支持
        if len(self._multiple_gpus) > 1:
            print("Multiple GPUs")
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # 核心训练
        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        """训练调度器。

        第一个任务: 直接训练 (func + rd)
        后续任务: 先检测是否需要扩展，如果需要则先训练新 Adapter
        """
        self._network.to(self._device)

        if self._cur_task == 0:
            # 第一个任务: func 阶段 + rd 阶段
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f"{total_params:,} total parameters.")
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad
            )
            print(f"{total_trainable_params:,} training parameters.")
            self._train_new(train_loader, test_loader)
        else:
            # ---- 扩展检测 ----
            # 开启检测模式，让所有 SEMA 模块计算 Z-score
            for module in self._network.backbone.modules():
                if isinstance(module, SEMAModules):
                    module.detecting_outlier = True

            detect_loader = DataLoader(
                train_loader.dataset,
                batch_size=self.args["detect_batch_size"],
                shuffle=True, num_workers=num_workers,
            )
            added = self._detect_outlier(detect_loader, train_loader, test_loader, 0)

            # 关闭检测模式
            for module in self._network.backbone.modules():
                if isinstance(module, SEMAModules):
                    module.detecting_outlier = False

            # 如果没有触发扩展，直接训练（直接复用已有 Adapter 的路由组合）
            if added == 0:
                self.update_optimizer_and_scheduler(
                    num_epoch=self.args["func_epoch"], lr=self.init_lr
                )
                self._init_train(
                    self.args["func_epoch"], train_loader, test_loader,
                    self.optimizer, self.scheduler, phase="func",
                )

        # 任务结束: 冻结旧 Adapter
        for module in self._network.backbone.modules():
            if isinstance(module, SEMAModules):
                module.end_of_task_training()

    def _train_new(self, train_loader, test_loader):
        """训练新添加的 Adapter。

        包含两个阶段:
        1. func 阶段: 训练功能适配器 + 路由器 + 分类器（交叉熵损失）
        2. rd 阶段: 训练表征描述器（重建损失）
        """
        # func 阶段
        self.update_optimizer_and_scheduler(
            num_epoch=self.args["func_epoch"], lr=self.init_lr
        )
        self._init_train(
            self.args["func_epoch"], train_loader, test_loader,
            self.optimizer, self.scheduler, phase="func",
        )

        # rd 阶段
        self.update_rd_optimizer_and_scheduler(
            num_epoch=self.args["rd_epoch"], lr=self.args["rd_lr"]
        )
        self._init_train(
            self.args["rd_epoch"], train_loader, test_loader,
            self.rd_optimizer, self.rd_scheduler, phase="rd",
        )

    def _detect_outlier(self, detect_loader, train_loader, test_loader, added):
        """离群检测 + 扩展。

        递归检测: 每次检测到需要扩展就添加 Adapter、
        训练新 Adapter、冻结，然后继续检测剩余数据。

        Args:
            detect_loader: 检测数据加载器
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            added: 已经添加的 Adapter 数（递归追踪）

        Returns:
            int: 本次检测中添加的 Adapter 总数
        """
        is_added = False

        for i, (_, inputs, targets) in enumerate(detect_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            model_outcome = self._network(inputs)
            added_record = model_outcome["added_record"]

            # 如果任何一层触发了扩展 (added == True)
            if sum(added_record) > 0:
                added += 1
                is_added = True

                # 关闭检测模式，准备训练新 Adapter
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.detecting_outlier = False

                # 训练新添加的 Adapter
                self._train_new(train_loader, test_loader)

                # 冻结新训练的 Adapter
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.freeze_functional()
                        module.freeze_rd()
                        module.reset_newly_added_status()

                # 重新开启检测模式
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.detecting_outlier = True

        # 递归: 如果本次检测中有扩展，继续检测剩余数据
        if is_added:
            return self._detect_outlier(
                detect_loader, train_loader, test_loader, added
            )
        else:
            return added

    def _init_train(self, total_epoch, train_loader, test_loader,
                    optimizer, scheduler, phase="func"):
        """单阶段训练循环。

        使用 LossTracker 自动记录每个 epoch 的损失到 wandb。

        Args:
            total_epoch: 训练 epoch 数
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            phase: "func" = 功能适配器训练, "rd" = 表征描述器训练
        """
        tracker = self.get_loss_tracker()
        prog_bar = tqdm(range(total_epoch))

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outcome = self._network(inputs)

                logits = outcome["logits"]
                logits = logits[:, : self._total_classes]

                # 后续任务: 将旧类的 logits 设为 -inf（只让模型在新类上做选择）
                if self._cur_task > 0:
                    logits[:, : self._known_classes] = -float("inf")

                # 根据阶段选择不同的损失函数
                if phase == "func":
                    # 功能阶段: 交叉熵分类损失
                    loss = F.cross_entropy(logits, targets)
                elif phase == "rd":
                    # RD 阶段: 表征描述器的重建损失
                    loss = outcome["rd_loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 记录损失到 LossTracker
                tracker.update(**{phase: loss.item()})

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            # 每个 epoch 结束时上报损失到 wandb
            avg_losses = tracker.flush(epoch)
            avg_loss = avg_losses.get(phase, 0)

            # 每 5 个 epoch 验证一次
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = (
                "{} Task {}, Epoch {}/{} => Loss {:.3f}, "
                "Train_accy {:.2f}, Test_accy {:.2f}".format(
                    phase, self._cur_task, epoch + 1, total_epoch,
                    avg_loss, train_acc, test_acc,
                )
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _eval_cnn(self, loader):
        """CNN 评估（重写父类以处理 SEMA 的字典输出）。"""
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outcome = self._network(inputs)
                logits = outcome["logits"]
                outputs = logits[:, : self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _compute_accuracy(self, model, loader):
        """简单准确率计算（重写以处理 SEMA 的字典输出）。"""
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outcome = self._network(inputs)
                logits = outcome["logits"]
                outputs = logits[:, : self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    # ====== 优化器管理 ======

    def update_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        """创建 func 阶段的优化器和调度器。

        只优化:
        - functional 模块的参数（新 Adapter 的功能部分）
        - router 模块的参数（新 Router 列）
        - fc 层的参数（分类器头）
        - vpt 参数（如果使用）
        """
        lr = self.args["init_lr"] if lr is None else lr
        func_params = [
            p for n, p in self._network.named_parameters()
            if (
                "functional" in n or "router" in n or "fc" in n or "vpt" in n
            )
            and p.requires_grad
        ]
        if self.args["optimizer"] == "sgd":
            self.optimizer = optim.SGD(
                func_params, momentum=0.9, lr=lr,
                weight_decay=self.args["weight_decay"],
            )
        elif self.args["optimizer"] == "adam":
            self.optimizer = optim.AdamW(
                func_params, lr=lr, weight_decay=self.args["weight_decay"],
            )

        min_lr = self.args["min_lr"] if self.args["min_lr"] is not None else 1e-8
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epoch, eta_min=min_lr,
        )

    def update_rd_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        """创建 rd 阶段的优化器和调度器。

        只优化: rd 模块的参数（新表征描述器的编码器和解码器）
        """
        lr = self.args["rd_lr"] if lr is None else lr
        rd_params = [
            p for n, p in self._network.named_parameters()
            if "rd" in n and p.requires_grad
        ]
        if self.args["optimizer"] == "sgd":
            self.rd_optimizer = optim.SGD(
                rd_params, momentum=0.9, lr=lr,
                weight_decay=self.args["weight_decay"],
            )
        elif self.args["optimizer"] == "adam":
            self.rd_optimizer = optim.AdamW(
                rd_params, lr=lr, weight_decay=self.args["weight_decay"],
            )

        min_lr = self.args["min_lr"] if self.args["min_lr"] is not None else 1e-8
        self.rd_scheduler = (
            optim.lr_scheduler.CosineAnnealingLR(
                self.rd_optimizer, T_max=num_epoch, eta_min=min_lr,
            )
            if self.rd_optimizer
            else None
        )

    def save_checkpoint(self, filename):
        """保存 SEMA 检查点（只保存 adapter 和 fc 参数）。"""
        state_dict = self._network.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if "adapter" in k or ("fc" in k and "block" not in k):
                save_dict[k] = v
        torch.save(save_dict, "{}.pth".format(filename))

    def load_checkpoint(self, filename):
        """加载 SEMA 检查点。"""
        self._network.load_state_dict(torch.load(filename), strict=False)
