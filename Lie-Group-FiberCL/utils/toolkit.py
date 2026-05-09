"""
持续学习通用工具函数
====================
提供参数统计、张量转换、准确率计算等基础功能。
"""

import os
import numpy as np
import torch


def count_parameters(model, trainable=False):
    """统计模型参数数量。

    Args:
        model: PyTorch 模型
        trainable: 若为 True，只统计 require_grad=True 的参数

    Returns:
        int: 参数总数
    """
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    """将 PyTorch 张量转换为 numpy 数组。"""
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    """将类别标签转换为 one-hot 编码。

    Args:
        targets: [N] 标签张量
        n_classes: 类别总数

    Returns:
        [N, n_classes] one-hot 张量
    """
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    """递归创建目录（如果不存在）。"""
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, init_cls=10, increment=10):
    """计算持续学习的分组准确率。

    将类别按任务分组，分别计算每组准确率，同时计算旧类和新类的准确率。
    - 旧类准确率 (old): 衡量模型的稳定性（是否遗忘）
    - 新类准确率 (new): 衡量模型的可塑性（是否学会）

    Args:
        y_pred: [N] top-1 预测
        y_true: [N] 真实标签
        nb_old: 本次任务之前已经见过的类别数
        init_cls: 第一个任务的类别数
        increment: 每个后续任务新增的类别数

    Returns:
        dict: {
            "total": 总准确率,
            "00-09": 第一组准确率,
            "10-19": 第二组准确率, ...,
            "old": 旧类准确率,
            "new": 新类准确率
        }
    """
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}

    # 总体准确率
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # 初始类组准确率（第一个任务的所有类）
    idxes = np.where(np.logical_and(y_true >= 0, y_true < init_cls))[0]
    label = "{}-{}".format(str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0"))
    all_acc[label] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    # 后续每个增量任务的组准确率
    for class_id in range(init_cls, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # 旧类准确率（衡量遗忘程度）
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # 新类准确率（衡量可塑性）
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def split_images_labels(imgs):
    """将 ImageFolder 的 imgs 列表拆分为图像路径和标签数组。

    Args:
        imgs: ImageFolder.imgs 返回的 (path, label) 列表

    Returns:
        (images_array, labels_array): numpy 数组
    """
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])
    return np.array(images), np.array(labels)
