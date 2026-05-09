"""
持续学习数据集类
===============

为每个数据集定义 iData 子类，封装:
- 数据下载/加载
- 训练/测试数据增强（transform）
- 类别排序

通过 lif_cl.paths.resolve_data_path() 自动定位数据集路径。
"""

import os
import sys
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lif_cl.paths import resolve_data_path


class iData(object):
    """数据集基类。"""
    train_trsf = []     # 训练数据增强
    test_trsf = []      # 测试数据增强
    common_trsf = []    # 共用的归一化等变换
    class_order = None  # 类别排序


def build_transform(is_train, args):
    """构建数据增强变换。

    训练模式: RandomResizedCrop + RandomHorizontalFlip → ToTensor
    测试模式: Resize + CenterCrop → ToTensor
    """
    input_size = 224
    resize_im = input_size > 32

    if is_train:
        scale = (0.05, 1.0)
        ratio = (3.0 / 4.0, 4.0 / 3.0)
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(transforms.Resize(size, interpolation=3))
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    return t


class iCIFAR100(iData):
    """CIFAR-100 数据集（224×224 版本）。

    用于 SEMA 等需要 ViT 输入尺寸的方法。
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = []
        self.class_order = np.arange(100).tolist()

    def download_data(self):
        """下载/加载 CIFAR-100 数据。"""
        data_path = resolve_data_path("cifar100")
        train_dataset = datasets.cifar.CIFAR100(
            data_path, train=True, download=True
        )
        test_dataset = datasets.cifar.CIFAR100(
            data_path, train=False, download=True
        )
        self.train_data = train_dataset.data
        self.train_targets = np.array(train_dataset.targets)
        self.test_data = test_dataset.data
        self.test_targets = np.array(test_dataset.targets)


class iCIFAR10(iData):
    """CIFAR-10 数据集（224×224 版本）。"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = []
        self.class_order = np.arange(10).tolist()

    def download_data(self):
        """下载/加载 CIFAR-10 数据。"""
        data_path = resolve_data_path("cifar10")
        train_dataset = datasets.cifar.CIFAR10(
            data_path, train=True, download=True
        )
        test_dataset = datasets.cifar.CIFAR10(
            data_path, train=False, download=True
        )
        self.train_data = train_dataset.data
        self.train_targets = np.array(train_dataset.targets)
        self.test_data = test_dataset.data
        self.test_targets = np.array(test_dataset.targets)


class iImageNetR(iData):
    """ImageNet-R 数据集。

    包含 200 类，是 ImageNet 的变体（rendition）。
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = []
        self.class_order = np.arange(200).tolist()

    def download_data(self):
        """加载 ImageNet-R 数据。"""
        data_path = resolve_data_path("imagenet-r")
        train_dir = os.path.join(data_path, "train")
        test_dir = os.path.join(data_path, "test")
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    """ImageNet-A 数据集。

    包含 200 类，是 ImageNet 的对抗性变体。
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = []
        self.class_order = np.arange(200).tolist()

    def download_data(self):
        """加载 ImageNet-A 数据。"""
        data_path = resolve_data_path("imagenet-a")
        train_dir = os.path.join(data_path, "train")
        test_dir = os.path.join(data_path, "test")
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iVTAB(iData):
    """VTAB 数据集。"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = []
        self.class_order = np.arange(50).tolist()

    def download_data(self):
        """加载 VTAB 数据。"""
        data_path = resolve_data_path("vtab")
        train_dir = os.path.join(data_path, "train")
        test_dir = os.path.join(data_path, "test")
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
