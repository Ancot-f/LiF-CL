"""
CODA-Prompt: COntinual Decomposed Attention-based Prompt for Continual Learning.

CODA-Prompt 是一种基于注意力分解的提示学习方法, 用于无需回放的持续学习.
核心创新:
  - Gram-Schmidt 正交化 prompt 组件, 确保组件间多样性.
  - 可学习注意力权重对不同 prompt 组件加权求和.
  - prompt_loss 稀疏性正则化, 鼓励每个任务仅使用少数组件.

参考论文:
  - Smith et al., "CODA-Prompt: COntinual Decomposed Attention-based Prompting
    for Rehearsal-Free Continual Learning", CVPR 2023.

核心特点:
  - 仅训练 prompt 参数和 FC 层, 冻结 ViT 骨干.
  - 自定义 CosineSchedule 余弦学习率调度.
  - 类别加权交叉熵损失缓解类别不均衡.
  - 第一个 session 用 VPT 微调, 之后切换 simple shot.
"""

import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.optim import Optimizer
import math
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import CodaPromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

# 第一个 session 使用 VPT 微调模型, 之后切换为 simple shot
num_workers = 8


class Learner(BaseLearner):
    """
    CODA-Prompt 学习者.

    通过 Gram-Schmidt 正交化 + 注意力加权实现 prompt 组件分解,
    稀疏性正则化实现任务间的组件专业化.

    关键属性:
      - dw_k: 每类数据权重, 用于类别平衡的加权交叉熵.
      - CodaPromptVitNet: 含 Gram-Schmidt 正交化 prompt 的网络.
    """

    def __init__(self, args):
        """
        初始化 CODA-Prompt 学习者.

        Args:
            args: 配置参数字典.
        """
        super().__init__(args)

        # CODA-Prompt 专用网络 (含 Gram-Schmidt 正交化)
        self._network = CodaPromptVitNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args

        # 统计可训练参数: 仅 fc 和 prompt, ViT 骨干完全冻结
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.fc.parameters() if p.requires_grad) + sum(p.numel() for p in self._network.prompt.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} fc and prompt training parameters.')


    def after_task(self):
        """每个任务完成后的回调: 更新已知类别计数."""
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        """
        执行增量任务训练.

        流程:
          1. process_task_count: 通知 prompt 模块新任务到来, 扩展注意力组件.
          2. 构建 DataLoader (drop_last=True 确保 batch 一致).
          3. 训练.

        Args:
            data_manager: 数据管理器.
        """
        self._cur_task += 1

        # 非首个任务: 通知 prompt 处理任务计数, 可能扩展注意力组件
        if self._cur_task > 0:
            try:
                if self._network.module.prompt is not None:
                    self._network.module.prompt.process_task_count()
            except:
                if self._network.prompt is not None:
                    self._network.prompt.process_task_count()

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        self.data_weighting()
        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def data_weighting(self):
        """初始化类别权重. 当前使用均匀权重 (全1), 可扩展为逆频率加权."""
        self.dw_k = torch.tensor(np.ones(self._total_classes + 1, dtype=np.float32))
        self.dw_k = self.dw_k.to(self._device)

    def get_optimizer(self):
        """构造优化器, 仅优化 prompt 和 fc 参数 (ViT 骨干冻结)."""
        # 收集 prompt 和 fc 的可训练参数
        if len(self._multiple_gpus) > 1:
            params = list(self._network.module.prompt.parameters()) + list(self._network.module.fc.parameters())
        else:
            params = list(self._network.prompt.parameters()) + list(self._network.fc.parameters())
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(params, momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(params, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = CosineSchedule(optimizer, K=self.args["tuned_epoch"])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """
        CODA-Prompt 主训练循环.

        核心损失: loss = loss_supervised + prompt_loss.sum()
          - loss_supervised: 类别加权的交叉熵损失.
          - prompt_loss: prompt 注意力稀疏性正则化 (鼓励每个任务使用少数组件).

        旧类别 logits 设为 -inf, 避免旧类别干扰当前任务训练.

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
            optimizer: 优化器.
            scheduler: 学习率调度器.
        """
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # 前向传播: 返回 logits + prompt_loss (稀疏性正则化)
                logits, prompt_loss = self._network(inputs, train=True)
                logits = logits[:, :self._total_classes]

                # 遮蔽旧类别, 仅关注新类别
                logits[:, :self._known_classes] = float('-inf')
                # 获取每样本的类别权重
                dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
                # 类别加权的交叉熵损失
                loss_supervised = (F.cross_entropy(logits, targets.long()) * dw_cls).mean()

                # 总损失 = 分类损失 + prompt 稀疏性正则化
                loss = loss_supervised + prompt_loss.sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _eval_cnn(self, loader):
        """
        评估: 返回 top-k 预测和真实标签.

        Args:
            loader: 数据加载器.

        Returns:
            tuple: (y_pred [N, topk], y_true [N,])
        """
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # 评估时 train=False, 不使用随机性
                outputs = self._network(inputs)[:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)[:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


class _LRScheduler(object):
    """
    自定义学习率调度器基类.

    实现状态管理和基础步进逻辑. 子类需实现 get_lr().
    """
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """返回调度器状态字典 (不含优化器)."""
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """加载调度器状态字典."""
        self.__dict__.update(state_dict)

    def get_lr(self):
        """子类实现: 返回每参数组的学习率."""
        raise NotImplementedError

    def step(self, epoch=None):
        """执行一步学习率更新."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CosineSchedule(_LRScheduler):
    """
    CODA-Prompt 专用的余弦学习率调度.

    公式: base_lr * cos(99*pi*epoch / (200*(K-1)))
    该特殊余弦函数在论文中被证明优于标准余弦调度.
    """

    def __init__(self, optimizer, K):
        self.K = K
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        """
        计算当前学习率.

        Args:
            base_lr: 基础学习率.

        Returns:
            float: 当前学习率.
        """
        return base_lr * math.cos((99 * math.pi * (self.last_epoch)) / (200 * (self.K-1)))

    def get_lr(self):
        """为所有参数组计算当前学习率."""
        return [self.cosine(base_lr) for base_lr in self.base_lrs]