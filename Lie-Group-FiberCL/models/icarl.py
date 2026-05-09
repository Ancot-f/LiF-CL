"""
iCaRL: Incremental Classifier and Representation Learning.

iCaRL 是最经典的类增量学习 (Class-Incremental Learning) 方法之一. 它通过
三个核心机制来缓解灾难性遗忘:
  1. 回放内存 (Rehearsal Memory): 存储旧类别的代表性样例, 在新任务训练时
     混入当前 batch 中一起训练.
  2. 知识蒸馏 (Knowledge Distillation): 使用旧模型 (固定) 的输出作为软标签,
     在旧类别上对新模型进行蒸馏, 保持模型对旧知识的记忆.
  3. 最近类均值分类器 (Nearest-Mean-of-Exemplars, NME): 使用每个类别的
     样例均值作为原型, 通过最近邻搜索进行分类 (推理阶段可选).

参考论文:
  - Rebuffi et al., "iCaRL: Incremental Classifier and Representation Learning",
    CVPR 2017.
    https://arxiv.org/abs/1611.07725

核心特点:
  - 任务0: 标准交叉熵训练 (初始化阶段).
  - 任务>0: 交叉熵 (分类损失) + KD (蒸馏损失) 联合训练.
  - 使用 MultiStepLR 学习率调度.
  - 训练完成后更新回放内存 (herding selection).
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
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8
num_workers = 8


class Learner(BaseLearner):
    """
    iCaRL 学习者.

    实现增量分类器与表示学习的经典算法: 回放 + 蒸馏 + NME 分类.

    关键属性:
      - _old_network: 上一任务训练后冻结的网络副本, 用于知识蒸馏.
      - T: 蒸馏温度参数, 控制软标签的平滑度.
    """

    def __init__(self, args):
        """
        初始化 iCaRL 学习者.

        Args:
            args: 配置参数字典.
        """
        super().__init__(args)
        self._network = IncrementalNet(args, True)

    def after_task(self):
        """
        每个任务完成后的回调.

        1. 将当前网络深拷贝并冻结, 作为下一任务的旧网络 (用于蒸馏).
        2. 更新已知类别计数.
        3. 记录回放内存大小.
        """
        # 深拷贝当前网络并冻结所有参数, 作为知识蒸馏的教师模型
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        """
        执行增量任务训练.

        流程与 finetune 类似, 但增加了:
          - 回放内存混入训练数据 (appendent).
          - 加载旧网络到设备用于蒸馏.
          - 训练后更新回放内存.

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

        # 构建训练 DataLoader, 混入回放内存中的旧类别样例
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        # 构建测试 DataLoader
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        # 训练完成后更新回放内存 (herding selection)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        """
        训练调度器.

        - task 0: 初始化训练 (仅交叉熵损失).
        - task > 0: 增量训练 (交叉熵 + 知识蒸馏).

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
        """
        self._network.to(self._device)
        # 将旧的冻结网络也移至设备 (用于蒸馏)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            # ---- 初始任务: 标准监督训练 ----
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
            # ---- 增量任务: 交叉熵 + KD 损失 ----
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
        初始任务训练循环.

        仅使用交叉熵损失, 不使用任何增量学习技巧.

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
            optimizer: 优化器.
            scheduler: 学习率调度器.
        """
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
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
        增量任务训练循环.

        核心损失: loss = loss_clf + loss_kd
          - loss_clf: 交叉熵分类损失 (所有类别, 包括回放样本).
          - loss_kd: 知识蒸馏损失, 让新模型在旧类别上的输出分布接近旧模型,
            从而保留旧知识.

        蒸馏损失公式 (KL 散度):
          KD_loss = -sum(softmax(old_logits/T) * log_softmax(new_logits/T)) / N

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
            optimizer: 优化器.
            scheduler: 学习率调度器.
        """
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                # 分类损失: 标准交叉熵
                loss_clf = F.cross_entropy(logits, targets)
                # 知识蒸馏损失: 仅在旧类别上计算
                # _KD_loss 计算新旧模型在旧类别 logits 上的 KL 散度
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    self.args["T"],
                )

                # 总损失 = 分类损失 + 蒸馏损失
                loss = loss_clf + loss_kd

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
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)


def _KD_loss(pred, soft, T):
    """
    知识蒸馏损失 (KL 散度).

    让新模型的预测分布接近旧模型的输出分布, 从而将旧知识迁移到新模型.

    步骤:
      1. 对 logits 除以温度 T, 软化概率分布.
      2. 预测端取 log_softmax, 目标端取 softmax.
      3. 计算两者的交叉熵 (等价于 KL 散度).

    Args:
        pred: 新模型在旧类别上的 logits.
        soft: 旧模型 (冻结) 在旧类别上的 logits.
        T: 蒸馏温度, 值越大分布越平滑.

    Returns:
        torch.Tensor: 蒸馏损失值 (标量).
    """
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
