"""
CoIL: Continual Invariant Learning — 基于最优传输的持续不变学习
==============================================================

CoIL 通过最优传输 (Optimal Transport) 理论在任务之间建立特征空间的对齐,
从而缓解类增量学习中的灾难性遗忘.

核心思想:
  1. 最优传输规划 (OT Plan): 使用 Sinkhorn 算法求解当前任务类别与下一任务
     类别之间的最优传输映射矩阵 T.
  2. 分类头初始化: 利用 OT 映射矩阵 T 将旧类别 FC 权重变换为新类别的初始权重.
  3. 两阶段训练:
      - 阶段1 (epoch<1): OT 初始化引导, 鼓励新分支输出接近 OT 初始化分布.
      - 阶段2 (epoch>=1): OT 协同调优, 通过反向 OT 对齐新旧特征空间.
  4. 使用余弦归一化特征和权重, 增强特征判别性.

参考论文:
  - Co^2L: Contrastive Continual Learning (ICCV 2021)
  - Sinkhorn Distances (Cuturi, NeurIPS 2013)

核心特点:
  - Sinkhorn 算法求解离散 OT, 用于分类头初始化和训练正则化.
  - 权重范数校准 (calibration_term) 保持变换前后范数一致.
  - 使用回放内存 (rehearsal memory) 存储旧类别样例.
"""

import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import (
    IncrementalNet,
    CosineIncrementalNet,
    SimpleCosineIncrementalNet,
)
from utils.toolkit import target2onehot, tensor2numpy
import ot  # POT (Python Optimal Transport) 库, 用于 Sinkhorn 算法
from torch import nn
import copy

EPSILON = 1e-8


class Learner(BaseLearner):
    """
    CoIL (Continual Invariant Learning) 学习者.

    通过最优传输在任务间建立特征对齐, 缓解灾难性遗忘.

    关键属性:
      - _ot_prototype_means: 各类别的原型均值, 用于 OT 代价矩阵计算.
      - _ot_new_branch: OT 映射生成的新分支初始化权重.
      - _ot_old_branch: 反向 OT 映射生成的旧分支参考权重.
      - sinkhorn_reg: Sinkhorn 熵正则化系数 (越大映射越平滑).
      - calibration_term: 权重范数校准系数 gamma.
      - lamda: 旧类别占比 = _known_classes/_total_classes, 用于损失平衡.
    """

    def __init__(self, args):
        """
        初始化 CoIL 学习者.

        Args:
            args: 配置字典, 含 sinkhorn, calibration_term, norm_term, reg_term.
        """
        super().__init__(args)
        # 使用余弦归一化的增量网络 (特征和权重经 L2 归一化)
        self._network = SimpleCosineIncrementalNet(args, True)
        self.data_manager = None
        # 下周期新类别 FC 的 OT 初始化权重
        self.nextperiod_initialization = None
        self.sinkhorn_reg = args["sinkhorn"]
        self.calibration_term = args["calibration_term"]
        self.epochs = args["epochs"]
        self.args = args

    def after_task(self):
        """
        每个任务完成后的回调.

        1. 求解 OT 为下一任务生成分类头初始化权重.
        2. 深拷贝并冻结当前网络作为蒸馏教师.
        3. 更新已知类别计数.
        """
        # 求解最优传输, 为下一任务生成分类头初始化
        self.nextperiod_initialization = self.solving_ot()
        # 冻结当前网络作为下一任务的知识蒸馏教师
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def solving_ot(self):
        """
        使用最优传输计算下一任务分类头的初始化权重.

        流程:
          1. 提取已知类别和下一批类别的原型均值.
          2. 计算余弦距离代价矩阵 Q.
          3. Sinkhorn 算法求解 OT 矩阵 T.
          4. W_new = T^T @ W_old, 并校准范数.

        Returns:
            torch.Tensor: 变换后的新类别 FC 初始化权重, 或 None (训练结束).
        """
        with torch.no_grad():
            # 全部类别已训练完毕, 无需继续 OT
            if self._total_classes == self.data_manager.nb_classes:
                print("training over, no more ot solving")
                return None
            each_time_class_num = self.data_manager.get_task_size(1)
            # 提取所有已见类别及下一批类别的原型均值
            self._extract_class_means(
                self.data_manager, 0, self._total_classes + each_time_class_num
            )
            former_class_means = torch.tensor(
                self._ot_prototype_means[: self._total_classes]
            )
            next_period_class_means = torch.tensor(
                self._ot_prototype_means[
                    self._total_classes : self._total_classes + each_time_class_num
                ]
            )
            # 计算旧类别与新类别之间的余弦距离代价矩阵
            Q_cost_matrix = torch.cdist(
                former_class_means, next_period_class_means, p=self.args["norm_term"]
            )
            # ---- Sinkhorn 求解最优传输 ----
            # 均匀分布作为源和目标分布
            _mu1_vec = (
                torch.ones(len(former_class_means)) / len(former_class_means) * 1.0
            )
            _mu2_vec = (
                torch.ones(len(next_period_class_means)) / len(former_class_means) * 1.0
            )
            # 熵正则化的最优传输
            T = ot.sinkhorn(_mu1_vec, _mu2_vec, Q_cost_matrix, self.sinkhorn_reg)
            T = torch.tensor(T).float().to(self._device)
            # 用 OT 矩阵将旧 FC 权重变换为新 FC 权重初始化
            transformed_hat_W = torch.mm(
                T.T, F.normalize(self._network.fc.weight, p=2, dim=1)
            )
            # ---- 权重范数校准 ----
            # 确保变换后权重范数与旧权重一致, 避免 scale 漂移
            oldnorm = torch.norm(self._network.fc.weight, p=2, dim=1)
            newnorm = torch.norm(
                transformed_hat_W * len(former_class_means), p=2, dim=1
            )
            meannew = torch.mean(newnorm)
            meanold = torch.mean(oldnorm)
            gamma = meanold / meannew
            self.calibration_term = gamma
            self._ot_new_branch = (
                transformed_hat_W * len(former_class_means) * self.calibration_term
            )
        return transformed_hat_W * len(former_class_means) * self.calibration_term

    def solving_ot_to_old(self):
        """
        反向最优传输: 从新类别到旧类别的 OT 映射.

        用于协同调优阶段: 将新类别特征映射回旧类别空间,
        通过 OT 督促新旧特征对齐. 方向与 solving_ot 相反.

        Returns:
            torch.Tensor: 反向 OT 变换后的旧分支参考权重.
        """
        current_class_num = self.data_manager.get_task_size(self._cur_task)
        # 提取包含回放内存的类原型均值
        self._extract_class_means_with_memory(
            self.data_manager, self._known_classes, self._total_classes
        )
        former_class_means = torch.tensor(
            self._ot_prototype_means[: self._known_classes]
        )
        next_period_class_means = torch.tensor(
            self._ot_prototype_means[self._known_classes : self._total_classes]
        )
        # 代价矩阵: 新类别 -> 旧类别 (方向相反)
        Q_cost_matrix = (
            torch.cdist(
                next_period_class_means, former_class_means, p=self.args["norm_term"]
            )
            + EPSILON
        )  # EPSILON 防止数值错误
        _mu1_vec = torch.ones(len(former_class_means)) / len(former_class_means) * 1.0
        _mu2_vec = (
            torch.ones(len(next_period_class_means)) / len(former_class_means) * 1.0
        )
        T = ot.sinkhorn(_mu2_vec, _mu1_vec, Q_cost_matrix, self.sinkhorn_reg)
        T = torch.tensor(T).float().to(self._device)
        # 将新类别 FC 权重通过反向 OT 映射到旧类别空间
        transformed_hat_W = torch.mm(
            T.T,
            F.normalize(self._network.fc.weight[-current_class_num:, :], p=2, dim=1),
        )
        return transformed_hat_W * len(former_class_means) * self.calibration_term

    def incremental_train(self, data_manager):
        """
        执行增量任务训练.

        流程:
          1. 使用 OT 初始化权重扩展 FC 层.
          2. 构建 DataLoader (混入回放内存).
          3. 训练后更新回放内存.

        Args:
            data_manager: 数据管理器.
        """
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )

        # 扩展 FC 层, 用 OT 计算的初始化填充新类别部分
        self._network.update_fc(self._total_classes, self.nextperiod_initialization)
        self.data_manager = data_manager

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        # lamda = 旧类别占比, 用于平衡蒸馏和分类损失
        self.lamda = self._known_classes / self._total_classes
        # ---- 构建 DataLoader ----
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=4
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4
        )

        self._train(self.train_loader, self.test_loader)
        # 更新回放内存 (herding selection)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def _train(self, train_loader, test_loader):
        """
        训练入口: 准备优化器和调度器, 执行训练循环.

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
        """
        self._network.to(self._device)
        # 将旧网络 (教师) 也移至设备, 用于知识蒸馏
        if self._old_network is not None:
            self._old_network.to(self._device)
        optimizer = optim.SGD(
            self._network.parameters(), lr=self.args["lrate"], momentum=0.9, weight_decay=5e-4
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
        )
        self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        """
        CoIL 增量训练循环 — 两阶段 OT 引导训练.

        阶段1 (epoch<1): OT 初始化引导
          loss = lamda*distill + (1-lamda)*clf + 0.001*w_init*new_branch_distill
          通过 new_branch_distill 鼓励新分类头输出接近 OT 初始化分布.

        阶段2 (epoch>=1): OT 协同调优
          loss = lamda*distill + (1-lamda)*clf + reg*w_co*old_branch_distill
          通过反向 OT 映射 (new->old) 对齐新旧特征空间.

        OT 权重调度:
          - weight_ot_init = max(1-(epoch/2)^2, 0): 初期大, 逐渐衰减.
          - weight_ot_co_tuning = (epoch/epochs)^2: 初期小, 逐渐增大.

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
            optimizer: SG 优化器.
            scheduler: MultiStepLR 学习率调度器.
        """
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            # OT 正则项权重随时间变化
            weight_ot_init = max(1.0 - (epoch / 2) ** 2, 0)
            weight_ot_co_tuning = (epoch / self.epochs) ** 2.0

            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                output = self._network(inputs)
                logits = output["logits"]
                onehots = target2onehot(targets, self._total_classes)

                # 分类损失 (所有类别)
                clf_loss = F.cross_entropy(logits, targets)
                if self._old_network is not None:
                    # ---- 知识蒸馏损失 (KL 散度) ----
                    old_logits = self._old_network(inputs)["logits"].detach()
                    hat_pai_k = F.softmax(old_logits / self.args["T"], dim=1)
                    log_pai_k = F.log_softmax(
                        logits[:, : self._known_classes] / self.args["T"], dim=1
                    )
                    distill_loss = -torch.mean(torch.sum(hat_pai_k * log_pai_k, dim=1))

                    if epoch < 1:
                        # ---- 阶段1: OT 初始化引导 ----
                        # 鼓励新分类头输出接近 OT 映射的参考分布
                        features = F.normalize(output["features"], p=2, dim=1)
                        current_logit_new = F.log_softmax(
                            logits[:, self._known_classes :] / self.args["T"], dim=1
                        )
                        new_logit_by_wnew_init_by_ot = F.linear(
                            features, F.normalize(self._ot_new_branch, p=2, dim=1)
                        )
                        new_logit_by_wnew_init_by_ot = F.softmax(
                            new_logit_by_wnew_init_by_ot / self.args["T"], dim=1
                        )
                        new_branch_distill_loss = -torch.mean(
                            torch.sum(
                                current_logit_new * new_logit_by_wnew_init_by_ot, dim=1
                            )
                        )

                        loss = (
                            distill_loss * self.lamda
                            + clf_loss * (1 - self.lamda)
                            + 0.001 * (weight_ot_init * new_branch_distill_loss)
                        )
                    else:
                        # ---- 阶段2: OT 协同调优 ----
                        # 通过反向 OT 映射, 鼓励旧类别特征与 OT 对齐
                        features = F.normalize(output["features"], p=2, dim=1)
                        # 每30步更新一次反向 OT 映射以节省计算
                        if i % 30 == 0:
                            with torch.no_grad():
                                self._ot_old_branch = self.solving_ot_to_old()
                        old_logit_by_wold_init_by_ot = F.linear(
                            features, F.normalize(self._ot_old_branch, p=2, dim=1)
                        )
                        old_logit_by_wold_init_by_ot = F.log_softmax(
                            old_logit_by_wold_init_by_ot / self.args["T"], dim=1
                        )
                        old_branch_distill_loss = -torch.mean(
                            torch.sum(hat_pai_k * old_logit_by_wold_init_by_ot, dim=1)
                        )
                        loss = (
                            distill_loss * self.lamda
                            + clf_loss * (1 - self.lamda)
                            + self.args["reg_term"]
                            * (weight_ot_co_tuning * old_branch_distill_loss)
                        )
                else:
                    # 首个任务: 仅分类损失
                    loss = clf_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            # CoIL 每个 epoch 都测试, 因为 OT 正则项变化较大
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.epochs,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _extract_class_means(self, data_manager, low, high):
        self._ot_prototype_means = np.zeros(
            (data_manager.nb_classes, self._network.feature_dim)
        )
        with torch.no_grad():
            for class_idx in range(low, high):
                data, targets, idx_dataset = data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1),
                    source="train",
                    mode="test",
                    ret_data=True,
                )
                idx_loader = DataLoader(
                    idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4
                )
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)
                class_mean = class_mean / (np.linalg.norm(class_mean))
                self._ot_prototype_means[class_idx, :] = class_mean
        self._network.train()

    def _extract_class_means_with_memory(self, data_manager, low, high):

        self._ot_prototype_means = np.zeros(
            (data_manager.nb_classes, self._network.feature_dim)
        )
        memoryx, memoryy = self._data_memory, self._targets_memory
        with torch.no_grad():
            for class_idx in range(0, low):
                idxes = np.where(
                    np.logical_and(memoryy >= class_idx, memoryy < class_idx + 1)
                )[0]
                data, targets = memoryx[idxes], memoryy[idxes]
                # idx_dataset=TensorDataset(data,targets)
                # idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                _, _, idx_dataset = data_manager.get_dataset(
                    [],
                    source="train",
                    appendent=(data, targets),
                    mode="test",
                    ret_data=True,
                )
                idx_loader = DataLoader(
                    idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4
                )
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)
                class_mean = class_mean / np.linalg.norm(class_mean)
                self._ot_prototype_means[class_idx, :] = class_mean

            for class_idx in range(low, high):
                data, targets, idx_dataset = data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1),
                    source="train",
                    mode="test",
                    ret_data=True,
                )
                idx_loader = DataLoader(
                    idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4
                )
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)
                class_mean = class_mean / np.linalg.norm(class_mean)
                self._ot_prototype_means[class_idx, :] = class_mean
        self._network.train()
