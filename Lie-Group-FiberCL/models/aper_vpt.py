"""
APER-VPT: APER 框架的 Visual Prompt Tuning (VPT) 变体.

使用 VPT 实现参数高效持续学习: 在 ViT 输入空间插入可学习的 Prompt Token
(支持 Deep 和 Shallow 两种模式), 冻结主干网络, 只训练 Prompt Token.

参考: Jia et al., "Visual Prompt Tuning", ECCV 2022.

核心特点:
  - 冻结 ViT 骨干所有参数, 仅训练 prompt token 和分类头.
  - Deep VPT: 每层 Transformer 前插入新的 prompt token.
  - Shallow VPT: 仅在第一层前插入 prompt token.
  - 第一个 session 训练 VPT 参数, 之后用类原型替换 FC.
  - 训练后构造双分支网络进行原型分类.
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

# 第一个 session 训练 VPT (prompt token), 之后使用 SimpleCIL
num_workers = 8


class Learner(BaseLearner):
    """
    APER-VPT 学习者.

    使用 Visual Prompt Tuning 实现参数高效微调.
    冻结 ViT 骨干, 仅训练 input space 中的 prompt token.
    """
    def __init__(self, args):
        """初始化. 要求 backbone_type 包含 'vpt'."""
        super().__init__(args)

        if 'vpt' not in args["backbone_type"]:
            raise NotImplementedError('VPT requires VPT backbone')
    
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
        
        model = model.eval()
        embedding_list = []
        label_list = []
        # data_list=[]
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

        class_list=np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
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
        训练入口. 仅在首个 session 训练 VPT 参数.

        VPT 模式下 prompt token 已设为 requires_grad=True,
        ViT 骨干参数由 backbone 内部冻结.
        """
        self._network.to(self._device)


        if self._cur_task == 0:

            # 统计可训练参数 (prompt token 已默认 requires_grad=True)
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')

            # if some parameters are trainable, print the key name and corresponding parameter number
            if total_params != total_trainable_params:
                for name, param in self._network.named_parameters():
                    if param.requires_grad:
                        print(name, param.numel())

            if self.args['optimizer']=='sgd':
                optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
            elif self.args['optimizer']=='adam':
                optimizer=optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
            # optimizer=optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
            scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)

            self._init_train(train_loader, test_loader, optimizer, scheduler)
            self.construct_dual_branch_network()
        else:
            pass
        
        self.replace_fc(train_loader_for_protonet, self._network, None)
            

    def construct_dual_branch_network(self):
        network = MultiBranchCosineIncrementalNet(self.args, True)
        network.construct_dual_branch_network(self._network)
        self._network=network.to(self._device)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
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

    