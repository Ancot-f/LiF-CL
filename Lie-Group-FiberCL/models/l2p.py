"""
Learning to Prompt (L2P) 用于持续学习 (Continual Learning).

L2P 是一种基于提示 (Prompt) 的持续学习方法, 通过维护一个可学习的提示池 (Prompt Pool)
来适应不同的任务. 核心思想是: 冻结预训练的 ViT (Vision Transformer) 骨干网络,
仅训练少量可学习的提示参数, 从而使模型能够在不遗忘旧知识的情况下学习新任务.

参考论文:
  - Wang et al., "Learning to Prompt for Continual Learning", CVPR 2022.
    https://arxiv.org/abs/2112.08654

核心特点:
  - 冻结 ViT 预训练权重, 只训练 prompt 参数.
  - 对每个输入图像, 通过查询 (query) 机制从提示池中选取 top-k 个提示.
  - 支持 shared_prompt_pool 和 shared_prompt_key 机制, 在新任务时复用上一个
    任务的提示参数和键作为初始化.
  - 使用交叉熵损失 + 可选 pull_constraint (减少 prompt 间余弦相似度).
  - 在第一个任务时使用 VPT 方式微调模型, 之后切换到 simple shot 模式.
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

# 第一个 session 使用 VPT 微调模型, 之后切换到简单模式
num_workers = 8


class Learner(BaseLearner):
    """
    L2P (Learning to Prompt) 学习者.

    通过提示池机制实现持续学习: 冻结 ViT 骨干, 仅训练少量 prompt 参数,
    每个任务从 prompt pool 中查询最相关的提示进行推理.

    参数说明:
      - prompt_pool: 是否使用提示池.
      - shared_prompt_pool: 是否在新任务时共享上一任务的 prompt 参数作为初始化.
      - shared_prompt_key: 是否在新任务时共享上一任务的 prompt 键作为初始化.
      - top_k: 每个输入查询 top-k 个提示.
      - pull_constraint: 是否施加 prompt 多样性约束 (减少相似度).
      - freeze: 需要冻结的 ViT 参数名列表 (如 blocks, patch_embed, cls_token).
      - reinit_optimizer: 是否在每个增量任务开始时重新初始化优化器.
    """

    def __init__(self, args):
        """
        初始化 L2P 学习者.

        加载 PromptVitNet 网络, 根据 freeze 参数冻结 ViT 骨干的部分参数,
        并统计可训练参数量用于日志输出.

        Args:
            args: 配置参数字典, 包含网络结构、训练参数等.
        """
        super().__init__(args)

        # PromptVitNet 是包装了 ViT + prompt 机制的网络
        self._network = PromptVitNet(args, True)

        # ---- 训练超参数初始化 ----
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args

        # ---- 冻结 ViT 预训练参数 ----
        # Freeze the parameters for ViT.
        if self.args["freeze"]:
            # 首先冻结 original_backbone 的全部参数
            for p in self._network.original_backbone.parameters():
                p.requires_grad = False

            # 然后根据 freeze 列表中指定的前缀冻结 backbone 中对应的参数
            # freeze 可以是 blocks, patch_embed, cls_token 等
            # freeze args.freeze[blocks, patch_embed, cls_token] parameters
            for n, p in self._network.backbone.named_parameters():
                if n.startswith(tuple(self.args["freeze"])):
                    p.requires_grad = False

        # ---- 统计并打印参数信息 ----
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        # 如果有可训练参数, 打印每个可训练参数的名称和数量
        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

    def after_task(self):
        """
        每个任务结束后的回调.

        更新 _known_classes 为当前总类别数.
        """
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        """
        执行单个增量任务的训练.

        流程:
          1. 更新任务计数和总类别数.
          2. 构建当前任务的训练/测试 DataLoader.
          3. 支持多 GPU (DataParallel).
          4. 调用 _train 执行训练.

        Args:
            data_manager: 数据管理器实例.
        """
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # ---- 构建数据加载器 ----
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        # ---- 多 GPU 支持 ----
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        """
        训练主流程.

        步骤:
          1. 获取优化器和学习率调度器.
          2. 如果不是第一个任务, 先初始化 prompt (从上一个任务复制).
          3. 如果 reinit_optimizer 开启, 重新构造优化器.
          4. 执行训练循环.

        Args:
            train_loader: 训练 DataLoader.
            test_loader: 测试 DataLoader.
        """
        self._network.to(self._device)

        # 获取优化器和学习率调度器
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        # 非首个任务: 初始化 prompt 参数 (从上一任务的 prompt 复制)
        if self._cur_task > 0:
            self._init_prompt(optimizer)

        # 如果设置为重新初始化优化器, 则在 prompt 初始化后重新构造
        if self._cur_task > 0 and self.args["reinit_optimizer"]:
            optimizer = self.get_optimizer()

        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def get_optimizer(self):
        """
        根据配置构造优化器.

        仅优化 requires_grad=True 的参数 (即仅训练 prompt 和不被冻结的部分).
        支持 sgd / adam / adamw 三种优化器.

        Returns:
            PyTorch 优化器实例.
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
        根据配置构造学习率调度器.

        支持 cosine / steplr / constant 三种调度策略.
        constant 表示不做学习率衰减.

        Args:
            optimizer: PyTorch 优化器.

        Returns:
            PyTorch 学习率调度器实例 (或 None).
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
        在增量任务开始时初始化 prompt 参数.

        当启用 shared_prompt_pool 时, 将上一任务的 prompt 参数复制到当前任务的
        prompt 槽位, 实现跨任务的参数复用.
        当启用 shared_prompt_key 时, 将上一任务的 prompt 键也复制到当前任务.

        这样可以利用前一个任务学到的 prompt 作为新任务的良好初始化,
        加速收敛并缓解遗忘.

        Args:
            optimizer: 当前优化器实例 (prompt 复制后需要更新优化器的参数引用).
        """
        args = self.args
        model = self._network.backbone
        task_id = self._cur_task

        # ---- Transfer previous learned prompt params to the new prompt ----
        # 将上一个任务学到的 prompt 参数复制到新任务对应的 prompt 槽位
        if args["prompt_pool"] and args["shared_prompt_pool"]:
            # 计算上一任务 prompt 在池中的起止索引
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            # 计算当前任务 prompt 的起止索引
            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            # 边界检查: 确保不超出 prompt 池的大小
            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))

                # 在不记录梯度的情况下进行参数复制
                with torch.no_grad():
                    model.prompt.prompt.grad.zero_()
                    # 将上一任务的 prompt 值复制到当前任务的 prompt 槽位
                    model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
                    # 更新优化器的参数组以包含修改后的参数
                    optimizer.param_groups[0]['params'] = model.parameters()

        # ---- Transfer previous learned prompt param keys to the new prompt ----
        # 类似地复制 prompt 键 (用于 query-key 匹配机制)
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
                model.prompt.prompt_key.grad.zero_()
                model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                optimizer.param_groups[0]['params'] = model.parameters()

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """
        L2P 的主训练循环.

        在每个 epoch 中:
          1. 设置 backbone 为训练模式, original_backbone 为评估模式 (冻结).
          2. 前向传播时传入 task_id 以从 prompt 池中查询对应的 prompt.
          3. 将旧类别的 logits 设为 -inf, 确保仅新类别参与损失计算.
          4. 计算交叉熵损失, 可选地施加减少 prompt 相似度的约束.
          5. 反向传播更新 prompt 参数.

        Args:
            train_loader: 训练数据 DataLoader.
            test_loader: 测试数据 DataLoader.
            optimizer: 优化器.
            scheduler: 学习率调度器.
        """
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            # backbone 设置为训练模式 (仅 prompt 参数可训练)
            self._network.backbone.train()
            # original_backbone 始终保持评估模式 (参数冻结)
            self._network.original_backbone.eval()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # 前向传播: 传入 task_id 以便 prompt pool 查询对应的 prompt
                # train=True 表示训练模式, prompt 选择会包含一定的随机性
                output = self._network(inputs, task_id=self._cur_task, train=True)
                # 截取当前已知类别范围内的 logits
                logits = output["logits"][:, :self._total_classes]
                # 将旧类别 (已学过的类别) 的 logits 设为 -inf
                # 这样在 softmax 时旧类别的概率为0, 模型仅关注新类别
                logits[:, :self._known_classes] = float('-inf')

                # 交叉熵损失 (仅在新类别上有效)
                loss = F.cross_entropy(logits, targets.long())
                # 可选的 pull_constraint: 减少不同 prompt 之间的余弦相似度
                # 鼓励 prompt 的多样性, 防止所有 prompt 趋于相同
                if self.args["pull_constraint"] and 'reduce_sim' in output:
                    loss = loss - self.args["pull_constraint_coeff"] * output['reduce_sim']

                # 反向传播与参数更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # 计算训练准确率
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            # 学习率调度步进
            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            # 每5个epoch进行一次测试评估
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
        评估模式: 返回 top-k 预测结果和真实标签.

        用于在测试集上计算 top-1 和 top-5 准确率.

        Args:
            loader: 测试数据 DataLoader.

        Returns:
            tuple: (y_pred [N, topk], y_true [N,]) 预测结果和真实标签.
        """
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # 推理时 task_id 用于选取对应的 prompt
                outputs = self._network(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            # 获取 top-k 个预测
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
        计算模型在指定数据加载器上的 top-1 准确率.

        Args:
            model: 待评估的模型.
            loader: 数据加载器.

        Returns:
            float: Top-1 准确率 (百分比).
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
