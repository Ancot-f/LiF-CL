"""
SimpleCIL: 简单的类增量学习基线 (Simple Class-Incremental Learning).

SimpleCIL 是一种极其简单的类增量学习方法:
  1. 在第一个任务上使用标准监督训练 (或微调) 模型.
  2. 对于后续任务, 不进行任何训练 (仅推理).
  3. 使用原型网络 (Prototype) 方式扩展分类头: 对新类别的训练数据进行特征提取,
     取每类的特征均值作为该类在 FC 层的权重向量, 从而实现零样本增长分类.

参考论文:
  - 作为增量学习的简单基线, 无特定单一论文对应.
  - 思想类似于 Prototypical Networks (Snell et al., NeurIPS 2017).

核心特点:
  - 仅训练第一个任务, 之后不再更新特征提取器.
  - 通过 replace_fc() 用类原型替代 FC 权重, 实现增量分类.
  - 极度简洁, 但特征提取器不具备持续适应能力.
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
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy


num_workers = 8
batch_size = 128


class Learner(BaseLearner):
    """
    SimpleCIL 学习者.

    仅在第一个任务执行标准训练, 后续任务使用类原型替换 FC 权重.
    不更新特征提取器参数, 不进行任何增量训练.

    关键方法:
      - replace_fc: 使用每类特征的均值 (原型) 替换 FC 层的权重.
    """

    def __init__(self, args):
        """
        初始化 SimpleCIL 学习者.

        使用 SimpleVitNet 作为骨干网络 (支持 ViT).

        Args:
            args: 配置参数字典.
        """
        super().__init__(args)
        # SimpleVitNet: 简单的 ViT 包装, 不包含 prompt 等增量组件
        self._network = SimpleVitNet(args, True)
        self.args = args

    def after_task(self):
        """
        每个任务后的回调.

        更新已知类别数.
        """
        self._known_classes = self._total_classes

    def replace_fc(self,trainloader, model, args):
        """
        用类原型替换 FC 层的权重.

        核心思路: 对训练集中的每个类别, 计算该类别所有样本的特征向量的均值
        (类原型), 然后将 FC 层中该类对应的权重行替换为该类原型.

        这样在推理时, FC 层的输出等价于输入特征与各类原型的余弦相似度
        (如果特征和权重都经过 L2 归一化).

        Args:
            trainloader: 新类别训练数据的 DataLoader (mode='test', 不打乱).
            model: 当前模型.
            args: 配置参数 (未使用).

        Returns:
            model: 更新 FC 权重后的模型.
        """
        model = model.eval()
        embedding_list = []
        label_list = []
        # 第一步: 提取所有新类别样本的特征向量
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                # 使用 backbone 提取特征 (不含分类头)
                embedding = model.backbone(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        # 第二步: 对每个类别, 计算类原型 (特征均值), 替换 FC 权重
        class_list = np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            # 找到属于当前类别的所有样本索引
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            # 计算类原型: 特征均值
            proto = embedding.mean(0)
            # 将 FC 层中该类对应的权重替换为类原型
            self._network.fc.weight.data[class_index] = proto
        return model


    def incremental_train(self, data_manager):
        """
        执行增量任务训练.

        流程:
          1. 扩展 FC 层以容纳新类别.
          2. 创建训练/测试 DataLoader.
          3. 如果是第一个任务: 执行标准训练 (_train); 否则跳过训练.

        Args:
            data_manager: 数据管理器.
        """
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # 训练 DataLoader (仅新类别)
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # 测试 DataLoader (所有已见类别)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # 用于原型网络的数据加载器: mode='test' 表示不打乱, 保证特征提取的一致性
        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        """
        训练入口.

        仅在第一个任务执行标准训练. 之后调用 replace_fc 更新 FC 权重.

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
            train_loader_for_protonet: 用于原型计算的 DataLoader (不打乱).
        """
        self._network.to(self._device)
        # 仅在首个任务时执行训练; 后续任务直接使用原型替换 FC 权重
        # 注意: 当前实现中第一个任务执行标准 _train (未在本类定义),
        # 实际上是依赖基类的 _init_train.
        # replace_fc 在每个任务都会调用, 用新类别的原型更新 FC 权重
        self.replace_fc(train_loader_for_protonet, self._network, None)
