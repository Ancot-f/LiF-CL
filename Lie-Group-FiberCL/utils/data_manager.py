"""
持续学习数据管理器
=================

负责:
- 数据集加载和类别排序
- 任务划分（按 init_cls 和 increment 分割类别）
- 为每个任务提供训练/测试数据集
"""

import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNetR, iImageNetA, iVTAB


class DataManager(object):
    """持续学习数据管理器。

    根据数据集名称加载数据，按类别排序，
    将类别划分为初始任务和后续增量任务。

    Args:
        dataset_name: 数据集名称 (cifar100, imagenetr, vtab, ...)
        shuffle: 是否打乱类别顺序
        seed: 随机种子
        init_cls: 第一个任务的类别数
        increment: 后续每个任务的新增类别数
        args: 全局参数字典
    """

    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args=None):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."

        # 计算每个任务的类别增量
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        """任务总数。"""
        return len(self._increments)

    @property
    def nb_classes(self):
        """总类别数。"""
        return len(self._class_order)

    def get_task_size(self, task):
        """返回第 task 个任务的类别数。"""
        return self._increments[task]

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
        """获取数据集。

        Args:
            indices: 类别索引列表
            source: "train" 或 "test"
            mode: "train"（训练增强）, "test"（测试增强）, "flip"（水平翻转增强）
            appendent: 可选的附加数据 (data, targets)
            ret_data: 是否返回原始数据

        Returns:
            DummyDataset 或 (data, targets, DummyDataset)
        """
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        # 选择数据增强模式
        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose([
                *self._test_trsf,
                transforms.RandomHorizontalFlip(p=1.0),
                *self._common_trsf,
            ])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        # 按类别选取数据
        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            data.append(class_data)
            targets.append(class_targets)

        # 附加数据（如回放内存中的样本）
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data = np.concatenate(data)
        targets = np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        """加载数据集并设置类别排序。

        1. 根据数据集名称创建对应的 iData 实例
        2. 下载/加载数据
        3. 打乱类别顺序（如果需要）
        4. 重新映射类别索引
        """
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        # 原始数据
        self._train_data = idata.train_data
        self._train_targets = idata.train_targets
        self._test_data = idata.test_data
        self._test_targets = idata.test_targets
        self.use_path = idata.use_path

        # 数据增强
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # 类别排序
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # 重新映射类别索引
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    @staticmethod
    def _select(x, y, low_range, high_range):
        """从数据中选取指定类别范围的样本。"""
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]


class DummyDataset(Dataset):
    """简单的 PyTorch 数据集包装器。

    将 numpy 数组包装为可被 DataLoader 加载的 Dataset。
    """

    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            # 从文件路径加载图像
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            # 从 numpy 数组加载图像
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        return idx, image, label


def _map_new_class_index(y, order):
    """将原始标签映射为新的类别顺序。"""
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    """根据数据集名称创建对应的 iData 实例。"""
    name = dataset_name.lower()
    if name == "cifar100":
        return iCIFAR100(args)
    elif name == "cifar10":
        return iCIFAR10(args)
    elif name == "imagenetr" or name == "imagenet_r":
        return iImageNetR(args)
    elif name == "imageneta" or name == "imagenet_a":
        return iImageNetA(args)
    elif name == "vtab":
        return iVTAB(args)
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """PIL 图像加载器。

    从文件路径加载 RGB 图像。
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
