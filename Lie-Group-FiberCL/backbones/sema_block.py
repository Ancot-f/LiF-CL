"""
SEMA 模块（适配器管理 + 路由 + 扩展检测）
========================================

这是 SEMA 的核心实现，包含两个类：

1. AdapterModule: 单个适配器单元
   - 功能适配器 (Adapter): 瓶颈 MLP，处理特征
   - 表征描述器 (AE): 自编码器，检测分布偏移
   - 运行统计 (Records): 维护重建误差的均值和方差

2. SEMAModules: 一层 ViT 中的所有适配器的管理器
   - 维护该层的适配器列表
   - 可扩展加权路由器（softmax 加权组合多个适配器输出）
   - 自扩展检测（Z-score 判断是否添加新适配器）
   - 冻结逻辑（训练后冻结旧适配器，保证不遗忘）

核心算法流程:
  1. 每个新任务的数据过一遍所有旧的表征描述器
  2. 计算重建误差的 Z-score
  3. 如果该层所有旧 RD 的 Z-score 都超过阈值 → 触发扩展
  4. 从顶层向下扫描，第一个触发扩展的层添加新适配器
  5. 每次最多加 1 个适配器（子线性增长）
"""

import torch
import torch.nn as nn
from typing import List
import copy
import logging
from backbones.sema_components import Adapter, AE, Records


class AdapterModule(nn.Module):
    """单个适配器单元。

    组合了三个子模块:
    - functional (Adapter): 功能适配器，处理特征用于分类
    - rd (AE): 表征描述器，自编码器用于分布偏移检测
    - rd_loss_record (Records): 维护该 RD 在训练集上的重建误差统计

    前向返回:
    - func_out: 适配器输出（用于分类）
    - rd_loss: 重建误差（用于扩展检测 + rd 阶段训练）
    - z_score: 当前输入的重建误差 Z-score（用于检测异常）

    Args:
        config: 全局配置
        adapter_id: 适配器标识符，如 "9.0"（第 9 层第 0 个适配器）
        writer: 日志写入器（可选）
    """

    def __init__(self, config, adapter_id, writer):
        super().__init__()
        self.config = config
        # 功能适配器（瓶颈 MLP）
        self.functional = Adapter(
            config, adapter_id, dropout=0.1, bottleneck=config.ffn_num,
            init_option=config.ffn_adapter_init_option,
            adapter_scalar=config.ffn_adapter_scalar,
            adapter_layernorm_option=config.ffn_adapter_layernorm_option,
        )
        layer_id = int(adapter_id.split('.')[0])
        # 只在指定层范围内使用表征描述器
        self.not_addition_layer = (
            layer_id < config.adapt_start_layer or layer_id > config.adapt_end_layer
        )
        if self.not_addition_layer:
            self.rd = None
        else:
            self.rd = AE(self.config)
        self.activation = nn.ReLU()
        self.newly_added = True  # 标记此适配器是否为本次任务新添加的
        self.adapter_id = adapter_id
        self.writer = writer
        self.rd_loss_record = Records(max_len=config.buffer_size)

    def forward(self, x):
        """前向传播。

        Returns:
            func_out: 功能适配器输出
            rd_loss: 重建误差标量（不可扩展层或无 RD 时为 0）
            z_score: Z-score（不可扩展层或无足够数据时为 0）
        """
        # 功能适配器输出
        func_out = self.functional(x)

        if self.not_addition_layer:
            # 浅层不使用 RD，直接返回
            rd_loss = torch.tensor(0., device=x.device)
            return func_out, rd_loss, torch.zeros_like(rd_loss)
        else:
            rd_loss = self.rd.compute_reconstruction_loss(x)

        # 计算重建误差的 Z-score
        z_score = self.get_z_score_deviation(rd_loss)

        # 训练模式下收集统计
        if self.training:
            self.add_z_score_record(rd_loss)

        return func_out, rd_loss, z_score

    def get_z_score_deviation(self, rd_loss):
        """计算重建误差的 Z-score。

        Z-score = (当前误差 - 历史均值) / 历史标准差

        用于判断当前输入是否偏离了该 RD 训练时的特征分布。
        Z-score 越大，偏离越严重。
        """
        mean, stddev = self.rd_loss_record.mean, self.rd_loss_record.stddev
        if not self.rd_loss_record.length > 2:
            return torch.zeros_like(rd_loss)
        z_score = (rd_loss - mean) / stddev
        z_score = torch.abs(z_score)
        return z_score

    def add_z_score_record(self, rd_loss):
        """将当前重建误差加入运行统计。"""
        self.rd_loss_record.add_record(rd_loss.detach().cpu())


class SEMAModules(nn.Module):
    """一层 ViT 中所有适配器的管理器。

    核心功能:
    1. 维护该层的适配器列表 (self.adapters)
    2. 可扩展路由器 (self.router): 用 softmax 加权组合多个适配器的输出
    3. 自扩展检测: 当所有旧 RD 都认为当前输入异常时，自动添加新适配器
    4. 冻结旧适配器: 训练后锁定，保证不遗忘

    路由机制:
        output = MLP(x) + Σ w_i · Adapter_i(x)
        w = softmax(router(x_mean))

    当需要添加新适配器时:
    1. 创建新的 AdapterModule
    2. 创建新的路由器列（初始权重为 0）
    3. 冻结旧的路由器列和旧的适配器

    Args:
        config: 全局配置
        layer_id: 该模块所在的 ViT 层索引
        writer: 日志写入器（可选）
    """

    def __init__(self, config, layer_id, writer):
        super().__init__()
        self.adapters: List[AdapterModule] = nn.ModuleList()
        self.config = config
        self.act_func = nn.ReLU()
        self.layer_id = layer_id
        self.writer = writer
        self.newly_added = True
        self.added_for_task = True
        self.adapt_start_layer = config.adapt_start_layer
        self.adapt_end_layer = config.adapt_end_layer

        # 初始化第一个适配器
        self.add_adapter(initialize=True)
        self.added_adapter = 0

        # 路由器：d_model → 适配器数量（用 softmax 加权）
        self.router = nn.Linear(config.d_model, 1)
        self.new_router = None  # 扩展时临时存放新路由器列
        self.detecting_outlier = False  # 是否处于离群检测模式

    @property
    def num_adapters(self):
        return len(self.adapters)

    def _device(self):
        """获取该模块所在的设备。

        优先级: router 的设备 > 第一个适配器的设备 > CPU
        用于动态创建新张量时确定正确的设备。
        """
        if hasattr(self, 'router') and self.router is not None:
            return self.router.weight.device
        if len(self.adapters) > 0:
            return next(self.adapters[0].parameters()).device
        return torch.device('cpu')

    def set_new_router(self):
        """创建新路由器列（初始化权重为 0）。

        当添加新适配器时调用：新列初始化为 0，
        避免新适配器立即影响输出，让训练逐步激活。
        """
        self.new_router = nn.Linear(self.config.d_model, 1).to(self._device())

    def fix_router(self):
        """将新路由器列合并到路由器中。

        训练完成后调用:
        1. 将旧路由器权重和新路由器权重拼接
        2. 旧列保持不变（已冻结），新列被纳入
        3. 新的合并后的路由器替换旧路由器
        """
        trained_router = nn.Linear(
            self.config.d_model, len(self.adapters)
        ).to(self._device())
        old_router = self.router
        weight = copy.deepcopy(old_router.weight.data)
        new_weight = copy.deepcopy(self.new_router.weight.data)
        weight = torch.cat([weight, new_weight])
        trained_router.weight = nn.Parameter(weight)
        bias = copy.deepcopy(old_router.bias.data)
        new_bias = copy.deepcopy(self.new_router.bias.data)
        bias = torch.cat([bias, new_bias])
        trained_router.bias = nn.Parameter(bias)
        self.router = trained_router
        self.new_router = None

    def add_adapter(self, initialize=False):
        """添加一个新适配器。

        Args:
            initialize: True = 首次初始化，False = 运行时扩展
        """
        adapter_id = f"{self.layer_id}.{len(self.adapters)}"
        new_adapter = AdapterModule(self.config, adapter_id, self.writer).to(
            self._device()
        )
        self.newly_added = True
        self.added_for_task = True
        self.adapters.append(new_adapter)
        if not initialize:
            self.set_new_router()
        logging.info(f"Adapter {adapter_id} added at block {self.layer_id}")

    def forward(self, x):
        """前向传播: 路由 + 加权组合 + 扩展检测。

        Args:
            x: [B, N, d_model] ViT 块输入

        Returns:
            dict: {
                "func_out": 功能适配器组合输出,
                "rd_loss": 重建误差（用于 rd 阶段训练）,
                "added": 是否在此 forward 中添加了新适配器
            }
        """
        rd_loss = torch.tensor(0., device=x.device)
        added = False

        # 浅层不进行扩展检测
        not_addition_layer = (
            self.layer_id < self.adapt_start_layer
            or self.layer_id > self.adapt_end_layer
        )

        if not_addition_layer:
            func_out, _, _ = self.adapters[-1](x)
        else:
            # 所有适配器前向传播
            func_outs, rd_losses, z_scores = [], [], []
            for adapter in self.adapters:
                func_out, rd_loss_i, z_score = adapter(x)
                func_outs.append(func_out)
                rd_losses.append(rd_loss_i)
                z_scores.append(z_score)

            func_outs = torch.stack(func_outs)  # [M, B, N, d]
            rd_losses = torch.stack(rd_losses)  # [M]
            z_scores = torch.stack(z_scores)    # [M]

            # ---- 扩展检测 ----
            # 条件：所有旧 RD 的 Z-score 都超过阈值，且当前任务尚未扩展
            addition_criteria = (
                z_scores.mean(dim=1).min() > self.config.exp_threshold
                and self.layer_id >= self.adapt_start_layer
                and self.layer_id <= self.adapt_end_layer
                and not self.added_for_task
                and self.detecting_outlier
            )

            if addition_criteria:
                # 触发扩展！
                self.add_adapter()
                out = {
                    "func_out": torch.zeros_like(func_outs[0]),
                    "rd_loss": torch.tensor(0., device=x.device),
                    "added": True,
                }
                return out
            else:
                # 路由器加权组合
                logits = self.router(x.mean(dim=1))  # [B, M]
                if self.new_router is not None:
                    new_logits = self.new_router(x.mean(dim=1))  # [B, 1]
                    logits = torch.cat([logits, new_logits], dim=1)  # [B, M+1]
                mask = torch.softmax(logits, dim=1)  # [B, M(+1)]

                # 加权求和: func_out = Σ mask_i * func_out_i
                func_out = (
                    func_outs * mask.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
                ).sum(dim=0)

                # rd_loss: 如果有新适配器，用最新的 RD 的损失；否则为 0
                if self.adapters[-1].newly_added:
                    rd_loss = rd_losses[-1].mean()
                else:
                    rd_loss = torch.tensor(0., device=x.device)

        out = {"func_out": func_out, "rd_loss": rd_loss, "added": added}
        return out

    # ====== 任务结束后的冻结操作 ======

    def end_of_task_training(self):
        """任务训练结束后的清理操作。

        1. 冻结所有旧功能适配器（防止遗忘）
        2. 冻结所有旧表征描述器（统计不再更新）
        3. 重置 added_for_task 标志
        """
        self.freeze_functional()
        self.freeze_rd()
        self.reset_newly_added_status()
        self.added_for_task = False

    def reset_newly_added_status(self):
        """重置 newly_added 标志。"""
        self.newly_added = False
        for adapter in self.adapters:
            adapter.newly_added = False

    def freeze_functional(self):
        """冻结所有适配器的功能模块。

        旧适配器的功能模块不再参与梯度更新，
        保证已学知识不被覆盖（抗遗忘的核心机制）。
        """
        adapter_ls = self.adapters
        for adapter in adapter_ls:
            for param in adapter.functional.parameters():
                param.requires_grad = False
                param._grad = None
        # 合并新路由器列
        if self.new_router is not None:
            self.fix_router()
        for param in self.router.parameters():
            param.requires_grad = False
            param._grad = None

    def freeze_rd(self):
        """冻结所有表征描述器。

        冻结后 RD 不再更新统计和梯度，
        作为固定的分布检测器保留。
        """
        adapter_ls = self.adapters
        for adapter in adapter_ls:
            if adapter.rd is not None:
                for param in adapter.rd.parameters():
                    param.requires_grad = False
                    param._grad = None
                adapter.rd_loss_record.updating = False
