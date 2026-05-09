"""
DER: Dynamically Expandable Representation for Class-Incremental Learning.

DER (Dynamic Expansion and Representation) 是一种通过动态扩展网络容量
来应对类增量学习的方法. 每当新任务到来时, 模型会扩展一个新的特征提取器
(backbone), 仅训练新扩展的部分, 旧的 backbone 参数被冻结.

注意: 当前实现仅包含动态扩展 (Dynamic Expansion) 部分, 原始论文中的
掩码 (masking) 和剪枝 (pruning) 机制未在此实现.

参考论文:
  - Yan et al., "DER: Dynamically Expandable Representation for Class
    Incremental Learning", CVPR 2021.
    https://arxiv.org/abs/2103.16788

核心特点:
  - 每个任务扩展一个独立的 backbone 分支, 旧分支冻结.
  - 使用辅助分类器 (aux_logits) 进行多任务监督.
  - 对旧类别的辅助分类标签设置为0 (背景类), 新类别映射到连续标签.
  - 使用回放内存 (rehearsal memory) 存储旧类别样例.
  - 训练损失 = 主分类损失 (所有类别) + 辅助分类损失 (新 backbone).
"""

import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import DERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

EPSILON = 1e-8
num_workers = 8


class Learner(BaseLearner):
    """
    DER (Dynamically Expandable Representation) 学习者.

    通过动态扩展独立 backbone 来应对新任务, 旧 backbone 冻结以防止遗忘.

    关键属性:
      - DERNet: 支持多 backbone 的网络结构.
      - 每个增量任务添加一个新的 backbone (backbones[-1]).
      - 辅助分类器 (aux_logits) 提供针对当前新 backbone 的监督信号.
    """

    def __init__(self, args):
        """
        初始化 DER 学习者.

        Args:
            args: 配置参数字典.
        """
        super().__init__(args)
        self._network = DERNet(args, True)

    def after_task(self):
        """
        每个任务完成后的回调.

        更新已知类别计���并记录回放内存大小.
        """
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        """
        执行增量任务训练.

        流程:
          1. 更新类别计数, 扩展 FC 层.
          2. 冻结所有旧的 backbone 参数 (backbones[0] 到 backbones[cur_task-1]).
          3. 构建训练/测试 DataLoader (混入回放样本).
          4. 执行训练.
          5. 更新回放内存.

        Args:
            data_manager: 数据管理器.
        """
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # 非首个任务: 冻结所有旧的 backbone 参数
        # 只有最新添加的 backbone (backbones[-1]) 可以训练
        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.backbones[i].parameters():
                    p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        # 构建训练 DataLoader, 混入回放内存
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        # 更新回放内存 (herding selection)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        """
        设置训练模式: 仅最后一个 backbone 为训练模式, 旧的 backbone 为 eval 模式.

        这确保旧的 BN 层统计信息不会被破坏.
        """
        self._network.train()
        if len(self._multiple_gpus) > 1 :
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        # 最新 backbone 设为训练模式
        self._network_module_ptr.backbones[-1].train()
        # 所有旧的 backbone 保持 eval 模式 (冻结 BN 等)
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network_module_ptr.backbones[i].eval()

    def _train(self, train_loader, test_loader):
        """
        训练调度器.

        - task 0: 初始训练 (标准交叉熵).
        - task > 0: 增量训练 (主分类损失 + 辅助分类损失).

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
        """
        self._network.to(self._device)
        if self._cur_task == 0:
            # ---- 初始任务训练 ----
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
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
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.args["lrate"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            # 增量训练使用主分类损失 + 辅助分类损失
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """
        初始任务训练循环 (第0个任务).

        仅使用标准交叉熵损失.

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
            optimizer: 优化器.
            scheduler: 学习率调度器.
        """
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        """
        增量任务训练循环 (第1个及以后的任务).

        核心损失 = loss_clf + loss_aux:
          - loss_clf: 主分类器的交叉熵损失 (所有已见类别).
          - loss_aux: 辅助分类器 (由新 backbone 产生) 的交叉熵损失.
            辅助分类器仅对新类别进行分类, 旧类别统一映射为标签0 (背景类).

        辅助分类器的作用: 为新 backbone 提供直接的监督信号, 确保新特征提取器
        能够有效学习当前任务的特征表示.

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
            optimizer: 优化器.
            scheduler: 学习率调度器.
        """
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_aux = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # 前向传播: 获取主分类 logits 和辅助分类 logits
                outputs = self._network(inputs)
                logits, aux_logits = outputs["logits"], outputs["aux_logits"]

                # 主分类损失 (所有类别)
                loss_clf = F.cross_entropy(logits, targets)

                # 辅助分类损失: 将旧类别映射为0, 新类别映射为连续正整数
                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes + 1 > 0,
                    aux_targets - self._known_classes + 1,
                    0,
                )
                loss_aux = F.cross_entropy(aux_logits, aux_targets)

                # 总损失 = 主分类 + 辅助分类
                loss = loss_clf + loss_aux

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_clf += loss_clf.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)
