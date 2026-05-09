"""
微调基线方法 (Simple Fine-tuning Baseline) 用于持续学习 (Continual Learning).

该方法是最简单的持续学习基线: 每个增量任务到来时, 直接在新任务数据上对模型进行
标准的监督微调训练, 不使用任何知识蒸馏、回放缓冲区或正则化技术.

参考论文:
  - 作为持续学习中最朴素的上界/下界基线使用, 无特定论文对应.
  - 常被用作衡量其他持续学习方法效果的对比基准.

核心特点:
  - 任务0: 在初始数据集上使用SGD优化器 + MultiStepLR调度器训练所有参数.
  - 任务>0: 在新任务数据上继续微调, 优化仅作用于分类头新增的部分 (通过
    将旧类别 logit 截断为 -inf 实现).
  - 无回放记忆、无知识蒸馏, 因此灾难性遗忘严重.
"""

import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

# DataLoader 使用的并行 worker 数量
num_workers = 8


class Learner(BaseLearner):
    """
    简单微调 (Fine-tune) 学习者.

    继承自 BaseLearner, 实现最简单的持续学习策略: 每个任务直接在新数据上
    进行监督训练, 不使用任何遗忘缓解技术.
    """

    def __init__(self, args):
        """
        初始化微调学习者.

        Args:
            args: 配置参数字典, 包含学习率、batch_size、epochs 等训练超参数.
        """
        super().__init__(args)
        # 使用 IncrementalNet 作为骨干网络, 支持增量式分类头扩展
        self._network = IncrementalNet(args, True)

    def after_task(self):
        """
        每个任务训练完成后的回调.

        更新 _known_classes 为当前总类别数, 为下一个任务的训练做准备.
        """
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        """
        执行单个增量任务的训练流程.

        步骤:
          1. 更新任务计数和总类别数.
          2. 扩展分类头 (fc 层) 以容纳新类别.
          3. 构建当前任务的训练/测试 DataLoader.
          4. 根据任务编号选择不同的训练策略 (初始训练 vs 增量训练).
          5. 若使用多 GPU, 通过 DataParallel 包裹网络.

        Args:
            data_manager: 数据管理器, 负责根据类别范围提供训练/测试数据集.
        """
        # 递增当前任务编号 (任务从0开始, 第一个任务 cur_task=1)
        self._cur_task += 1
        # 计算任务结束后的总类别数
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        # 扩展网络最后一层全连接层, 使其输出维度匹配总类别数
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # ---- 构建训练集 DataLoader ----
        # 仅获取当前任务新类别的训练数据
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        # ---- 构建测试集 DataLoader ----
        # 测试集包含从类别0到当前任务所有类别
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        # ---- 多 GPU 支持 ----
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        # 执行训练
        self._train(self.train_loader, self.test_loader)
        # 训练完成后从 DataParallel 中恢复原始模型
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        """
        训练调度器: 根据当前任务编号选择初始化训练或增量训练模式.

        - task 0 (cur_task == 1): 初始化训练, 使用较高的初始学习率.
        - task > 0: 增量训练, 使用较低的学习率, 且仅对当前新类别计算损失.

        Args:
            train_loader: 训练数据 DataLoader.
            test_loader: 测试数据 DataLoader.
        """
        self._network.to(self._device)
        if self._cur_task == 0:
            # ---- 初始任务训练 ----
            # 使用 SGD 优化器 + MultiStepLR 学习率调度
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            # ---- 增量任务训练 ----
            # 使用较小的学习率 (lrate), 避免过度扰动已学习的表示
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """
        初始任务的标准训练循环 (第0个任务).

        使用全类别的交叉熵损失进行训练, 不使用任何增量学习特殊处理.

        Args:
            train_loader: 训练数据 DataLoader.
            test_loader: 测试数据 DataLoader.
            optimizer: PyTorch 优化器实例.
            scheduler: PyTorch 学习率调度器实例.
        """
        # 获取损失追踪器, 用于记录和汇总训练过程中的损失值
        tracker = self.get_loss_tracker()
        # 进度条显示
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                # 数据转移到目标设备 (CPU/GPU)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # 前向传播: 获取分类 logits
                logits = self._network(inputs)["logits"]

                # 标准交叉熵损失
                loss = F.cross_entropy(logits, targets)
                # 反向传播与参数更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 记录损失值
                tracker.update(ce=loss.item())

                # 计算训练准确率
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            # 学习率调度步进
            scheduler.step()
            # 计算并格式化训练准确率
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            # 获取当前 epoch 的平均损失
            avg_losses = tracker.flush(epoch)
            avg_loss = avg_losses.get("ce", 0)

            # 每5个epoch进行一次测试评估
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    avg_loss,
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    avg_loss,
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        """
        增量任务的训练循环 (第1个及以后的任务).

        与初始训练不同, 此方法:
          - 仅计算新类别范围内的交叉熵损失 (fake_targets = targets - _known_classes).
          - logits 截断到新类别范围, 避免旧类别干扰当前任务训练.
          - 这实际上是一种简单的不使用回放/蒸馏的增量微调策略.

        Args:
            train_loader: 训练数据 DataLoader.
            test_loader: 测试数据 DataLoader.
            optimizer: PyTorch 优化器实例.
            scheduler: PyTorch 学习率调度器实例.
        """
        tracker = self.get_loss_tracker()
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                # 将真实标签映射到新类别的局部索引空间
                # 例如: 如果已有100个旧类别, 新类别从100开始, 则100 -> 0, 101 -> 1, ...
                fake_targets = targets - self._known_classes
                # 仅在新类别范围内计算交叉熵损失
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes :], fake_targets
                )

                loss = loss_clf

                # 反向传播与参数更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tracker.update(ce=loss.item())

                # 计算训练准确率 (基于完整 logits)
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_losses = tracker.flush(epoch)
            avg_loss = avg_losses.get("ce", 0)
            # 每5个epoch进行测试评估
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    avg_loss,
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    avg_loss,
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)
