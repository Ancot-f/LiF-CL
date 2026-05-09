"""
FOSTER: Feature Boosting and Compression for Class-Incremental Learning.

FOSTER 是一种两阶段类增量学习方法:
  1. Feature Boosting (特征增强阶段):
     为新任务扩展一个额外的特征提取器 (backbone), 通过分类损失、特征增强损失
     和知识蒸馏损失联合训练, 增强模型的表示能力.
  2. Feature Compression (特征压缩阶段):
     通过知识蒸馏将增强后的模型 (Teacher) 压缩回与原始模型相同大小的学生模型
     (Student), 去除冗余, 控制模型规模增长.

参考论文:
  - Wang et al., "FOSTER: Feature Boosting and Compression for Class-Incremental
    Learning", ECCV 2022.
    https://arxiv.org/abs/2204.04662

核心特点:
  - 每次增量任务扩展新的 backbone, 旧 backbone 冻结.
  - Boosting: 联合训练新 backbone + 分类头, 使用交叉熵 + 特征增强损失 + KD.
  - Compression: 通过双向知识蒸馏 (BKD) 将教师模型压缩为学生模型.
  - 使用类别平衡权重 (per-class weights) 缓解数据不均衡.
  - 支持 weight alignment 校准分类头对新旧类别的偏差.
  - 使用回放内存 (rehearsal memory) 存储旧类别样例.
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
from utils.inc_net import FOSTERNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

# 完整 FOSTER 实现请参考官方仓库: https://github.com/G-U-N/ECCV22-FOSTER

EPSILON = 1e-8


class Learner(BaseLearner):
    """
    FOSTER (Feature Boosting and Compression) 学习者.

    主要流程 (对新任务):
      1. Boosting: 扩展新 backbone, 联合训练.
      2. Teacher Weight Alignment (可选).
      3. Compression: 通过知识蒸馏压缩为单一模型 (学生网络).

    关键属性:
      - _snet: 压缩后的学生网络, 下一任务的输入.
      - beta1/beta2: 类别平衡权重的衰减因子.
      - lambda_okd: 知识蒸馏损失权重.
      - oofc: 旧类别分类头处理方式 ('az' 表示零初始化旧的, 'ft' 表示全微调).
    """

    def __init__(self, args):
        """
        初始化 FOSTER 学习者.

        Args:
            args: 配置参数字典, 包含 beta1, beta2, lambda_okd 等 FOSTER 特有参数.
        """
        super().__init__(args)
        self.args = args
        # FOSTERNet 支持多 backbone 扩展
        self._network = FOSTERNet(args, True)
        # _snet: 压缩后的学生网络, 初始为 None
        self._snet = None
        # 类别平衡重加权参数 (boosting阶段)
        self.beta1 = args["beta1"]
        # 类别平衡重加权参数 (compression阶段)
        self.beta2 = args["beta2"]
        # 每类权重张量
        self.per_cls_weights = None
        # 是否对教师模型进行 weight alignment
        self.is_teacher_wa = args["is_teacher_wa"]
        # 是否对学生模型进行 weight alignment
        self.is_student_wa = args["is_student_wa"]
        # 知识蒸馏损失的权重系数
        self.lambda_okd = args["lambda_okd"]
        # weight alignment 的缩放因子
        self.wa_value = args["wa_value"]
        # 旧类别分类头处理模式: 'az'=只允许旧的FC参数为零, 'ft'=全微调
        self.oofc = args["oofc"].lower()

    def after_task(self):
        """
        每个任务完成后的回调.

        更新已知类别数, 记录当前回放内存中的样例数量.
        """
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        """
        执行增量任务训练.

        流程:
          1. 如果不是第一个任务, 将模型替换为上一轮压缩后的学生网络 _snet.
          2. 更新 FC 层以容纳新类别.
          3. 冻结旧的 backbone 参数.
          4. 训练 -> 构建回放内存.

        Args:
            data_manager: 数据管理器.
        """
        self.data_manager = data_manager
        self._cur_task += 1
        # 第2个任务开始, 使用上一轮压缩后的学生网络作为基础
        if self._cur_task > 1:
            self._network = self._snet
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # 非首个任务: 冻结旧的 backbone 和老的 FC 层
        if self._cur_task > 0:
            for p in self._network.backbones[0].parameters():
                p.requires_grad = False
            for p in self._network.oldfc.parameters():
                p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        # 构建训练 DataLoader (包含回放内存中的旧类别数据)
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"],
            pin_memory=True,
        )
        # 构建测试 DataLoader
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args["num_workers"],
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        # 构建回放内存
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        """
        设置模型为训练模式 (仅最后一个 backbone + 可训练部分).

        旧 backbone 始终保持 eval 模式以冻结 BN 等统计信息.
        """
        self._network_module_ptr.train()
        self._network_module_ptr.backbones[-1].train()
        if self._cur_task >= 1:
            self._network_module_ptr.backbones[0].eval()

    def _train(self, train_loader, test_loader):
        """
        训练调度器.

        - task 0: 初始训练 (标准监督学习).
        - task > 0: 特征增强 (Feature Boosting) -> 权重对齐 (可选) -> 特征压缩 (Feature Compression).

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
        """
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if self._cur_task == 0:
            # ---- 初始任务: 标准训练 ----
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["init_epochs"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            # ---- 增量任务: Boosting + Compression ----

            # 第一步: 计算类别平衡权重 (基于有效样本数理论)
            # Effective Number of Samples: (1-beta^n) / (1-beta)
            # 其中 n 为该类样本数, beta 控制重加权程度
            cls_num_list = [self.samples_old_class] * self._known_classes + [
                self.samples_new_class(i)
                for i in range(self._known_classes, self._total_classes)
            ]

            effective_num = 1.0 - np.power(self.beta1, cls_num_list)
            per_cls_weights = (1.0 - self.beta1) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )

            logging.info("per cls weights : {}".format(per_cls_weights))
            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)

            # Boosting 阶段的优化器
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.args["lr"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["boosting_epochs"]
            )
            # 如果 oofc='az', 将新类别的 FC 权重初始化为零 (仅允许旧的FC非零)
            if self.oofc == "az":
                for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                    if i == 0:
                        p.data[
                            self._known_classes :, : self._network_module_ptr.out_dim
                        ] = torch.tensor(0.0)
            elif self.oofc != "ft":
                assert 0, "not implemented"
            # ---- 阶段1: 特征增强 (Feature Boosting) ----
            self._feature_boosting(train_loader, test_loader, optimizer, scheduler)

            # ---- Teacher Weight Alignment (可选) ----
            # 校准教师模型对新旧类别的 FC 权重偏差
            if self.is_teacher_wa:
                self._network_module_ptr.weight_align(
                    self._known_classes,
                    self._total_classes - self._known_classes,
                    self.wa_value,
                )
            else:
                logging.info("do not weight align teacher!")

            # 第二步: 使用 beta2 重新计算类别平衡权重 (用于压缩阶段)
            cls_num_list = [self.samples_old_class] * self._known_classes + [
                self.samples_new_class(i)
                for i in range(self._known_classes, self._total_classes)
            ]
            effective_num = 1.0 - np.power(self.beta2, cls_num_list)
            per_cls_weights = (1.0 - self.beta2) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )
            logging.info("per cls weights : {}".format(per_cls_weights))
            self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self._device)
            # ---- 阶段2: 特征压缩 (Feature Compression) ----
            self._feature_compression(train_loader, test_loader)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """
        初始任务的训练循环 (第0个任务).

        使用标准交叉熵损失进行监督训练, 不使用任何增量学习特殊技巧.

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
            optimizer: 优化器.
            scheduler: 学习率调度器.
        """
        prog_bar = tqdm(range(self.args["init_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
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
                    self.args["init_epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _feature_boosting(self, train_loader, test_loader, optimizer, scheduler):
        """
        FOSTER 阶段1: 特征增强 (Feature Boosting).

        为新任务训练一个新的 backbone (附加到 backbones 列表末尾), 同时:
          - 使用类别平衡的交叉熵损失 (loss_clf) 训练新分类头.
          - 使用特征增强损失 (loss_fe) 提升新 backbone 的表示能力.
          - 使用知识蒸馏损失 (loss_kd) 将旧模型的知识迁移到新模型,
            防止灾难性遗忘.

        总损失: loss = loss_clf + loss_fe + lambda_okd * loss_kd

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
            optimizer: 优化器.
            scheduler: 学习率调度器.
        """
        prog_bar = tqdm(range(self.args["boosting_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_fe = 0.0
            losses_kd = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                outputs = self._network(inputs)
                logits, fe_logits, old_logits = (
                    outputs["logits"],
                    outputs["fe_logits"],
                    outputs["old_logits"].detach(),  # 旧模型的 logits, 不传播梯度
                )
                # 类别平衡的交叉熵损失 (对主分类头)
                loss_clf = F.cross_entropy(logits / self.per_cls_weights, targets)
                # 特征增强损失 (对新 backbone 的特征输出)
                loss_fe = F.cross_entropy(fe_logits, targets)
                # 知识蒸馏损失 (旧类别上的 KD, 防止遗忘)
                loss_kd = self.lambda_okd * _KD_loss(
                    logits[:, : self._known_classes], old_logits, self.args["T"]
                )
                loss = loss_clf + loss_fe + loss_kd

                optimizer.zero_grad()
                loss.backward()
                # 如果 oofc='az', 强制旧类别的 FC 梯度为零 (不更新旧的FC权重)
                if self.oofc == "az":
                    for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                        if i == 0:
                            p.grad.data[
                                self._known_classes :,
                                : self._network_module_ptr.out_dim,
                            ] = torch.tensor(0.0)
                elif self.oofc != "ft":
                    assert 0, "not implemented"
                optimizer.step()
                losses += loss.item()
                losses_fe += loss_fe.item()
                losses_clf += loss_clf.item()
                # 按旧类别占比缩放 KD 损失用于日志记录
                losses_kd += (
                    self._known_classes / self._total_classes
                ) * loss_kd.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["boosting_epochs"],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fe {:.3f}, Loss_kd {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["boosting_epochs"],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fe / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _feature_compression(self, train_loader, test_loader):
        """
        FOSTER 阶段2: 特征压缩 (Feature Compression).

        目标: 将增强后的教师模型 (包含双 backbone) 压缩回单 backbone 的学生模型,
        控制模型规模不随任务数线性增长.

        流程:
          1. 创建学生网络 _snet (单 backbone).
          2. 将教师模型的第一个 backbone 权重复制给学生.
          3. 将教师的 oldfc 复制给学生.
          4. 使用 BKD (Bidirectional Knowledge Distillation) 训练学生,
             使学生模仿教师的输出分布.
          5. 可选地对学生进行 weight alignment.

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
        """
        # 创建学生网络
        self._snet = FOSTERNet(self.args, True)
        self._snet.update_fc(self._total_classes)
        if len(self._multiple_gpus) > 1:
            self._snet = nn.DataParallel(self._snet, self._multiple_gpus)
        if hasattr(self._snet, "module"):
            self._snet_module_ptr = self._snet.module
        else:
            self._snet_module_ptr = self._snet
        self._snet.to(self._device)
        # 复制教师模型的第一个 backbone 权重给学生
        self._snet_module_ptr.backbones[0].load_state_dict(
            self._network_module_ptr.backbones[0].state_dict()
        )
        # 复制旧 FC 层参数
        self._snet_module_ptr.copy_fc(self._network_module_ptr.oldfc)
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._snet.parameters()),
            lr=self.args["lr"],
            momentum=0.9,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.args["compression_epochs"]
        )
        # 教师模型设为 eval 模式
        self._network.eval()
        prog_bar = tqdm(range(self.args["compression_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._snet.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                dark_logits = self._snet(inputs)["logits"]
                # 从冻结的教师模型获取软标签
                with torch.no_grad():
                    outputs = self._network(inputs)
                    logits, old_logits, fe_logits = (
                        outputs["logits"],
                        outputs["old_logits"],
                        outputs["fe_logits"],
                    )
                # BKD 损失: 学生输出与教师输出之间的 KL 散度
                loss_dark = self.BKD(dark_logits, logits, self.args["T"])
                loss = loss_dark
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(dark_logits[: targets.shape[0]], dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._snet, test_loader)
                info = "SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["compression_epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["compression_epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)
        if len(self._multiple_gpus) > 1:
            self._snet = self._snet.module
        # ---- Student Weight Alignment (可选) ----
        if self.is_student_wa:
            self._snet.weight_align(
                self._known_classes,
                self._total_classes - self._known_classes,
                self.wa_value,
            )
        else:
            logging.info("do not weight align student!")

        # ---- 对学生网络进行评估 ----
        self._snet.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(self._device, non_blocking=True)
            with torch.no_grad():
                outputs = self._snet(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        cnn_accy = self._evaluate(y_pred, y_true)
        logging.info("darknet eval: ")
        logging.info("CNN top1 curve: {}".format(cnn_accy["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_accy["top5"]))

    @property
    def samples_old_class(self):
        """
        每个旧类别的回放样本数.

        如果使用固定内存 (_fixed_memory=True), 返回每类固定样本数;
        否则按总内存大小均分给所有已知类别.

        Returns:
            int: 每类回放样本数.
        """
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._known_classes

    def samples_new_class(self, index):
        """
        获取指定新类别的训练样本数.

        对于 CIFAR100 每个类固定 500 张图片, 其他数据集从 data_manager 获取.

        Args:
            index: 类别索引.

        Returns:
            int: 该类别的样本数.
        """
        if self.args["dataset"] == "cifar100":
            return 500
        else:
            return self.data_manager.getlen(index)

    def BKD(self, pred, soft, T):
        """
        双向知识蒸馏损失 (Bidirectional Knowledge Distillation).

        计算学生预测和教师软标签之间的加权 KL 散度.
        使用类别平衡权重对教师软标签进行重加权, 缓解类别不均衡问题.

        公式:
          pred = log_softmax(pred / T)
          soft = softmax(soft / T) * per_cls_weights (归一化)
          loss = -sum(soft * pred) / batch_size

        Args:
            pred: 学生模型的 logits.
            soft: 教师模型的 logits (软标签).
            T: 温度参数, 用于软化概率分布.

        Returns:
            torch.Tensor: BKD 损失值.
        """
        pred = torch.log_softmax(pred / T, dim=1)
        soft = torch.softmax(soft / T, dim=1)
        # 使用类别平衡权重对教师软标签进行重加权
        soft = soft * self.per_cls_weights
        soft = soft / soft.sum(1)[:, None]
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def _KD_loss(pred, soft, T):
    """
    标准知识蒸馏损失 (Kullback-Leibler Divergence).

    计算学生预测分布与教师软标签分布之间的 KL 散度.

    公式:
      pred = log_softmax(pred / T)
      soft = softmax(soft / T)
      loss = -sum(soft * pred) / batch_size

    Args:
        pred: 学生模型的 logits.
        soft: 教师模型的 logits (软标签).
        T: 温度参数.

    Returns:
        torch.Tensor: KD 损失值.
    """
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
