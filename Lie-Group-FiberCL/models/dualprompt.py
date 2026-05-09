"""
DualPrompt: 双提示池 (G-Prompt + E-Prompt) 用于持续学习.

DualPrompt 是 L2P 的改进版本, 引入了两种互补的提示类型:
  - G-Prompt (General Prompt): 跨任务共享的通用提示, 捕获任务间共性知识.
  - E-Prompt (Expert Prompt): 任务特定的专家提示, 从提示池中按任务查询.
  两者通过不同的插入位置 (Transformer 层) 共同作用, 使模型既能保留通用知识
  又能适应新任务的特性.

参考论文:
  - Wang et al., "DualPrompt: Complementary Prompting for Rehearsal-free
    Continual Learning", ECCV 2022.
    https://arxiv.org/abs/2204.04799

核心特点:
  - 双提示机制: G-Prompt 在浅层插入, E-Prompt 在深层插入或跨层共享.
  - 支持 use_prefix_tune_for_e_prompt 前缀微调模式.
  - 支持 shared_prompt_pool / shared_prompt_key 跨任务复用.
  - 冻结 ViT 预训练权重, 仅训练提示参数.
  - 使用交叉熵损失 + 可选 pull_constraint.
"""

import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import PromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

# 第一个 session 使用 VPT 微调模型, 之后切换为 simple shot
num_workers = 8


class Learner(BaseLearner):
    """
    DualPrompt 学习者.

    实现了双提示池策略: G-Prompt (任务通用) + E-Prompt (任务特定),
    同时通过 query-key 匹配机制从 E-Prompt 池中选择最相关的提示.

    与 L2P 的主要区别:
      - 增加了 G-Prompt: 在所有任务间共享, 提供通用知识的基线.
      - use_prefix_tune_for_e_prompt: 可选的前缀微调模式 (在每个 Transformer
        层前插入可学习的前缀 token).
    """

    def __init__(self, args):
        """
        初始化 DualPrompt 学习者.

        加载 PromptVitNet, 冻结 ViT 骨干参数, 统计可训练参数.

        Args:
            args: 配置参数字典.
        """
        super().__init__(args)

        self._network = PromptVitNet(args, True)

        # ---- 训练超参数 ----
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args

        # ---- 冻结 ViT 预训练参数 ----
        if self.args["freeze"]:
            for p in self._network.original_backbone.parameters():
                p.requires_grad = False

            for n, p in self._network.backbone.named_parameters():
                if n.startswith(tuple(self.args["freeze"])):
                    p.requires_grad = False

        # ---- 统计并打印参数数量 ----
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

    def after_task(self):
        """
        每个任务完成后的回调.

        更新已知类别计数.
        """
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        """
        执行增量任务训练.

        流程与 L2P 相同: 更新类别计数 -> 构建 DataLoader -> 训练.

        Args:
            data_manager: 数据管理器.
        """
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        """
        训练调度: 获取优化器和调度器 -> 初始化 prompt -> 执行训练.

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
        """
        self._network.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        if self._cur_task > 0:
            self._init_prompt(optimizer)

        if self._cur_task > 0 and self.args["reinit_optimizer"]:
            optimizer = self.get_optimizer()

        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def get_optimizer(self):
        """
        构造优化器, 仅优化可训练参数.

        Returns:
            PyTorch 优化器.
        """
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )

        return optimizer

    def get_scheduler(self, optimizer):
        """
        构造学习率调度器.

        Args:
            optimizer: PyTorch 优化器.

        Returns:
            学习率调度器 (或 None).
        """
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_prompt(self, optimizer):
        """
        初始化 E-Prompt 池参数 (从上一任务复制).

        与 L2P 的 _init_prompt 类似, 但操作的是 model.e_prompt (而非 model.prompt).
        当 use_prefix_tune_for_e_prompt=True 时, prompt 张量增加了一个维度
        (prefix token 数量), 切片索引需要额外包含这个维度.

        两种模式:
          1. shared_prompt_pool: 复制上一任务的 e_prompt 参数.
          2. shared_prompt_key: 复制上一任务的 e_prompt 键.

        Args:
            optimizer: 当前优化器, prompt 复制后需更新其参数引用.
        """
        args = self.args
        model = self._network.backbone
        task_id = self._cur_task

        # ---- 复制 E-Prompt 参数 (prompt values) ----
        if args["prompt_pool"] and args["shared_prompt_pool"]:
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                # 如果使用 prefix tune 模式, prompt 有额外的维度 (prefix长度)
                # 形状: [num_layers, prefix_len, num_prompts, dim]
                # 否则: [num_prompts, dim]
                cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args["use_prefix_tune_for_e_prompt"] else (slice(None), slice(cur_start, cur_end))
                prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args["use_prefix_tune_for_e_prompt"] else (slice(None), slice(prev_start, prev_end))

                with torch.no_grad():
                    model.e_prompt.prompt.grad.zero_()
                    model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                    optimizer.param_groups[0]['params'] = model.parameters()

        # ---- 复制 E-Prompt 键 (prompt keys for query-key matching) ----
        if args["prompt_pool"] and args["shared_prompt_key"]:
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))

            with torch.no_grad():
                model.e_prompt.prompt_key.grad.zero_()
                model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                optimizer.param_groups[0]['params'] = model.parameters()

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """
        DualPrompt 主训练循环.

        与 L2P 的本质相同: 冻结骨干, 仅训练 prompt 参数.
        通过 task_id 查询对应的 G-Prompt 和 E-Prompt, 旧类别 logits 置为 -inf.

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
            optimizer: 优化器.
            scheduler: 学习率调度器.
        """
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            self._network.original_backbone.eval()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # 前向传播: task_id 用于从 E-Prompt 池中查询任务特定提示
                output = self._network(inputs, task_id=self._cur_task, train=True)
                logits = output["logits"][:, :self._total_classes]
                # 遮蔽旧类别, 仅对新类别计算损失
                logits[:, :self._known_classes] = float('-inf')

                loss = F.cross_entropy(logits, targets.long())
                # 可选的 prompt 多样性约束: 减少 prompt 间相似度
                if self.args["pull_constraint"] and 'reduce_sim' in output:
                    loss = loss - self.args["pull_constraint_coeff"] * output['reduce_sim']

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
                outputs = self._network(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        """
        计算 top-1 准确率.

        Args:
            model: 待评估模型.
            loader: 数据加载器.

        Returns:
            float: Top-1 准确率 (%).
        """
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
