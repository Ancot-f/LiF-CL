"""
持续学习基础学习器 (BaseLearner)
===============================

提供所有持续学习方法共用的基础功能:
- 示例内存管理 (exemplar memory): 存储旧类样本用于回放
- NME 评估 (Nearest Mean of Exemplars): 基于类中心的最近邻分类
- 准确率评估: 分组准确率、旧类/新类准确率
- 与 lif_cl 集成: wandb 日志、loss 追踪、断点续跑

子类需实现:
- incremental_train(data_manager): 增量训练一个任务
- _train(...): 实际的训练循环
"""

import copy
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from utils.toolkit import tensor2numpy, accuracy

EPSILON = 1e-8  # 数值稳定性
batch_size = 64  # 示例加载器的默认 batch size


class BaseLearner(object):
    """持续学习基础类。

    子类（如 SEMA Learner）继承此类并实现具体的训练逻辑。

    Args:
        args: 参数字典，需包含:
            - memory_size: 示例内存大小
            - memory_per_class: 每类示例数
            - fixed_memory: 是否固定内存
            - device: 设备列表（如 [cuda:0]）
            - init_cls: 第一个任务的类别数
            - increment: 后续每个任务的类别增量
    """

    def __init__(self, args):
        self._cur_task = -1        # 当前任务索引（从 0 开始）
        self._known_classes = 0    # 已经学过的类别数
        self._total_classes = 0    # 当前见过的总类别数
        self._network = None       # 主干网络
        self._old_network = None   # 旧网络（用于知识蒸馏）
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5              # top-K 评估（默认 top-5）

        # 示例内存配置
        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self.args = args

        # lif_cl 集成
        self._wandb_logger = None
        self._loss_tracker = None

    # ====== lif_cl 集成 ======

    def set_wandb_logger(self, wandb_logger):
        """注入 wandb 日志器（由 trainer 调用）。"""
        self._wandb_logger = wandb_logger

    def get_loss_tracker(self, task_id=None):
        """获取或创建 LossTracker。

        每个任务自动创建一个新的 LossTracker，
        step 自动按 task_id * 1000 + epoch 计算。
        """
        if task_id is None:
            task_id = self._cur_task
        if self._loss_tracker is None or self._loss_tracker.task_id != task_id:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))
            from lif_cl.loss_tracker import LossTracker
            self._loss_tracker = LossTracker(self._wandb_logger, task_id=task_id)
        return self._loss_tracker

    # ====== 属性 ======

    @property
    def exemplar_size(self):
        """当前示例内存中的样本数。"""
        assert len(self._data_memory) == len(self._targets_memory), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        """每类的示例数。"""
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        """特征维度。"""
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    # ====== 示例内存管理 ======

    def build_rehearsal_memory(self, data_manager, per_class):
        """构建回放内存。"""
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def tsne(self, showcenters=False, Normalize=False):
        import umap
        import matplotlib.pyplot as plt
        print('now draw tsne results of extracted features.')
        tot_classes = self._total_classes
        test_dataset = self.data_manager.get_dataset(
            np.arange(0, tot_classes), source='test', mode='test')
        valloader = DataLoader(test_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=4)
        vectors, y_true = self._extract_vectors(valloader)
        if showcenters:
            fc_weight = self._network.fc.proj.cpu().detach().numpy()[:tot_classes]
            print(fc_weight.shape)
            vectors = np.vstack([vectors, fc_weight])

        if Normalize:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(vectors)

        if showcenters:
            clssscenters = embedding[-tot_classes:, :]
            centerlabels = np.arange(tot_classes)
            embedding = embedding[:-tot_classes, :]
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_true,
                              s=20, cmap=plt.cm.get_cmap("tab20"))
        plt.legend(*scatter.legend_elements())
        if showcenters:
            plt.scatter(clssscenters[:, 0], clssscenters[:, 1], marker='*',
                        s=50, c=centerlabels, cmap=plt.cm.get_cmap("tab20"),
                        edgecolors='black')

        plt.savefig(str(self.args['model_name']) + str(tot_classes) + 'tsne.pdf')
        plt.close()

    def _reduce_exemplar(self, data_manager, m):
        """减少已有示例（按每类 m 个）。"""
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0 else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0 else dt
            )

            # 计算类均值（用于 NME）
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        """为新类构建示例（herding 算法选择最有代表性的样本）。"""
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train", mode="test", ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Herding: 贪心选择最接近类均值的样本
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(exemplar_vectors, axis=0)
                mu_p = (vectors + S) / k
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(np.array(data[i]))
                exemplar_vectors.append(np.array(vectors[i]))
                vectors = np.delete(vectors, i, axis=0)
                data = np.delete(data, i, axis=0)

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0 else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0 else exemplar_targets
            )

            # 计算类均值
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        """统一构建示例（适用固定内存策略）。"""
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # 重新计算旧类的均值
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask], self._targets_memory[mask],
            )
            class_dset = data_manager.get_dataset(
                [], source="train", mode="test",
                appendent=(class_data, class_targets),
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            _class_means[class_idx, :] = mean

        # 为新类构建示例
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train", mode="test", ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Herding 选择
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(exemplar_vectors, axis=0)
                mu_p = (vectors + S) / k
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(np.array(data[i]))
                exemplar_vectors.append(np.array(vectors[i]))
                vectors = np.delete(vectors, i, axis=0)
                data = np.delete(data, i, axis=0)

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0 else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0 else exemplar_targets
            )

            # 计算类均值
            exemplar_dset = data_manager.get_dataset(
                [], source="train", mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            _class_means[class_idx, :] = mean

        self._class_means = _class_means

    # ====== 评估 ======

    def _evaluate(self, y_pred, y_true):
        """评估准确率。

        返回 top-1, top-K 和分组准确率。
        """
        ret = {}
        grouped = accuracy(
            y_pred.T[0], y_true, self._known_classes,
            self.args["init_cls"], self.args["increment"]
        )
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
        return ret

    def eval_task(self):
        """评估当前任务。"""
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def _eval_cnn(self, loader):
        """CNN 评估: 用分类器头预测。"""
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_nme(self, loader, class_means):
        """NME 评估: 基于类中心的最近邻分类。

        将测试样本分配给距离最近的类中心。
        """
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes]
        return np.argsort(scores, axis=1)[:, : self.topk], y_true

    def _extract_vectors(self, loader):
        """提取所有样本的特征向量。"""
        self._network.eval()
        vectors, targets = [], []
        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.module.extract_vector(_inputs.to(self._device))
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.extract_vector(_inputs.to(self._device))
                    )
                vectors.append(_vectors)
                targets.append(_targets)
        return np.concatenate(vectors), np.concatenate(targets)

    def _compute_accuracy(self, model, loader):
        """简单准确率计算（用于训练过程中的验证）。"""
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    # ====== 检查点 ======

    def save_checkpoint(self, filename):
        """保存模型检查点。"""
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        """任务完成后的回调（子类可重写）。"""
        pass

    def incremental_train(self, data_manager):
        """增量训练一个任务（子类必须实现）。"""
        pass

    def _train(self):
        """训练循环（子类必须实现）。"""
        pass

    def _get_memory(self):
        """获取回放内存数据。"""
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)
