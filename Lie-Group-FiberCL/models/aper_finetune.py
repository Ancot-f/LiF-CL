"""
APER-Finetune: APER 框架的全量微调基线.

APER (Adaptive Parameter-Efficient Regularization) 框架的基线变体之一.
在第一个 session 对 ViT/ResNet 进行全量微调, 之后切换到 SimpleCIL 模式
(冻结特征提取器, 用类原型替换 FC 权重).

注意: 全量微调会更新 ViT 所有参数, 参数量大, 但在持续学习设置中
仅作基线对比, 后续任务不做任何训练.

核心特点:
  - 首个 session 全量微调所有参数.
  - 后续任务使用 replace_fc() 以类原型替换 FC 权重.
  - 支持 resnet 和 vit 两种 backbone.
  - 训练后构造双分支网络 (MultiBranchCosineIncrementalNet).
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
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,MultiBranchCosineIncrementalNet,SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from timm.scheduler import create_scheduler

# 第一个 session 全量微调, 之后使用 SimpleCIL (原型分类)
num_workers = 8


class Learner(BaseLearner):
    """
    APER-Finetune 学习者.

    第一个 session 全量微调 ViT/ResNet, 后续任务用类原型替换 FC 权重.
    属于 APER 框架的全量微调基线.
    """
    def __init__(self, args):
        """
        初始化学习者. 根据 backbone_type 选择 ResNet 或 ViT 网络.

        Args:
            args: 配置字典.
        """
        super().__init__(args)
        if 'resnet' in args['backbone_type']:
            self._network = SimpleCosineIncrementalNet(args, True)
            self. batch_size=128
            self.init_lr=args["init_lr"] if args["init_lr"] is not None else  0.01
        else:
            self._network = SimpleVitNet(args, True)
            self. batch_size= args["batch_size"]
            self. init_lr=args["init_lr"]
        
        self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args=args

    def after_task(self):
        self._known_classes = self._total_classes
    
    def replace_fc(self,trainloader, model, args):
        """
        用类原型替换 FC 层权重.

        对每个类别, 计算该类所有样本的特征均值 (类原型),
        然后将 FC 层中该类对应的权重行替换为该原型.
        推理时 FC 输出等价于输入特征与各类原型的余弦相似度.

        Args:
            trainloader: 新类别训练数据 DataLoader (mode='test', 不打乱).
            model: 当前模型.
            args: 配置参数 (未使用).

        Returns:
            model: 更新 FC 权重后的模型.
        """
        model = model.eval()
        embedding_list = []
        label_list = []
        # 提取所有新类别样本的特征向量
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.to(self._device)
                label=label.to(self._device)
                embedding = model(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        # 对每个类别, 计算特征均值作为类原型, 替换 FC 权重
        class_list=np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]
            proto=embedding.mean(0)
            self._network.fc.weight.data[class_index]=proto
        return model

    
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        """
        训练入口.

        仅在第一个 session (cur_task==0) 执行全量微调训练.
        后续任务直接跳过训练, 仅用 replace_fc 更新 FC 权重.
        """
        self._network.to(self._device)
        if self._cur_task == 0:
            # 首个 session: 全量微调
            if self.args['optimizer']=='sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
            elif self.args['optimizer']=='adam':
                optimizer=optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
            scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            # 训练后构造双分支网络 (含余弦归一化 FC)
            self.construct_dual_branch_network()
        else:
            # 后续 session: 不训练, 跳过
            pass
        # 用类原型替换 FC 权重
        self.replace_fc(train_loader_for_protonet, self._network, None)


    def construct_dual_branch_network(self):
        """构造双分支余弦增量网络, 将单分支网络升级为支持余弦归一化的双分支结构."""
        network = MultiBranchCosineIncrementalNet(self.args, True)
        network.construct_dual_branch_network(self._network)
        self._network=network.to(self._device)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """
        初始任务标准训练循环.

        使用交叉熵损失进行全量监督训练.

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
            optimizer: 优化器.
            scheduler: 余弦学习率调度器.
        """
        prog_bar = tqdm(range(self.args['tuned_epoch']))
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
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    




