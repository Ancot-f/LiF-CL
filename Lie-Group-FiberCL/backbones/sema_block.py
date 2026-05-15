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
import torch.nn.functional as F
from typing import List
import copy
import math
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

        # ---- Bundle-aware 组件 (Group-Structured Positional Routing) ----
        # group_modulator: 根据群权重调制 adapter 输出 (关联向量丛的纤维方向修正)
        # geometry_rd: 检测群权重分布的偏移 (局部丛图有效性检测)
        self.has_bundle = getattr(config, 'use_group_pos', False)
        if self.has_bundle:
            num_groups = getattr(config, 'num_groups', 4)
            # Group Modulator: K → 2D (gamma, beta), 最后一层零初始化保证初始行为接近原始 SEMA
            modulator_hidden = max(num_groups * 2, config.d_model // 8)
            self.group_modulator = nn.Sequential(
                nn.Linear(num_groups, modulator_hidden),
                nn.GELU(),
                nn.Linear(modulator_hidden, 2 * config.d_model),
            )
            nn.init.zeros_(self.group_modulator[-1].weight)
            nn.init.zeros_(self.group_modulator[-1].bias)

            # Geometry RD: 检测群权重分布偏移
            if not self.not_addition_layer:
                geo_rd_dim = getattr(config, 'geo_rd_dim', max(2, min(8, num_groups // 2)))
                self.geo_rd_encoder = nn.Linear(num_groups, geo_rd_dim)
                self.geo_rd_decoder = nn.Linear(geo_rd_dim, num_groups)
                nn.init.kaiming_uniform_(self.geo_rd_encoder.weight, a=math.sqrt(5))
                nn.init.zeros_(self.geo_rd_encoder.bias)
                nn.init.kaiming_uniform_(self.geo_rd_decoder.weight, a=math.sqrt(5))
                nn.init.zeros_(self.geo_rd_decoder.bias)
                self.bundle_loss_record = Records(max_len=config.buffer_size)
            else:
                self.geo_rd_encoder = None
                self.geo_rd_decoder = None
                self.bundle_loss_record = None
        else:
            self.group_modulator = None
            self.geo_rd_encoder = None
            self.geo_rd_decoder = None
            self.bundle_loss_record = None

    def forward(self, x, group_info=None):
        """前向传播 (Bundle-aware 版本)。

        Bundle-aware 调制:
          group_weights → group_modulator → (gamma, beta)
          adapter_out = adapter_out * (1 + gamma) + beta
          这对应关联向量丛上纤维方向的群结构调制。

        Returns:
            func_out:     功能适配器输出 (经过 bundle-aware 调制)
            rd_loss:      特征重建误差 [B]
            geo_rd_loss:  群权重重建误差 [B]
            bundle_rd_loss: 联合 bundle 重建误差 [B]
            z_score:      特征 Z-score [B]
            bundle_z_score: bundle Z-score [B]
        """
        # 功能适配器输出 (Bottleneck MLP)
        func_out = self.functional(x)

        # ---- Bundle-aware Adapter 调制 (关联向量丛纤维修正) ----
        if group_info is not None and self.group_modulator is not None:
            group_weights = group_info["group_weights"]       # [B, K]
            gamma_beta = self.group_modulator(group_weights)  # [B, 2D]
            gamma, beta = gamma_beta.chunk(2, dim=-1)         # [B, D], [B, D]
            gamma = gamma.unsqueeze(1)  # [B, 1, D]
            beta = beta.unsqueeze(1)    # [B, 1, D]
            func_out = func_out * (1.0 + gamma) + beta

        # ---- 浅层: 无 RD, 直接返回 ----
        if self.not_addition_layer:
            zero = torch.tensor(0., device=x.device)
            return func_out, zero, zero, zero, zero, zero

        # ---- 特征 RD loss (原始 AE) ----
        rd_loss = self.rd.compute_reconstruction_loss(x)      # [B]
        z_score = self.get_z_score_deviation(rd_loss)

        # ---- Geometry RD loss (群权重分布检测) ----
        if group_info is not None and self.geo_rd_encoder is not None:
            geo_rd_loss = self._compute_geo_rd_loss(group_info["group_weights"])  # [B]
            lambda_geo = getattr(self.config, 'lambda_geo', 0.1)
            bundle_rd_loss = rd_loss + lambda_geo * geo_rd_loss
            bundle_z_score = self._get_bundle_z_score(bundle_rd_loss)
            if self.training:
                self.add_z_score_record(rd_loss)
                self._add_bundle_record(bundle_rd_loss)
        else:
            geo_rd_loss = torch.zeros_like(rd_loss)
            bundle_rd_loss = rd_loss
            bundle_z_score = z_score
            if self.training:
                self.add_z_score_record(rd_loss)

        return func_out, rd_loss, geo_rd_loss, bundle_rd_loss, z_score, bundle_z_score

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

    def _compute_geo_rd_loss(self, group_weights):
        """计算群权重重建损失 (Geometry RD loss).

        通过轻量自编码器检测群权重分布是否偏离历史分布。
        对应局部丛图有效性检测: 如果群权重不能被重建,
        说明当前输入的结构群选择模式与历史不一致。
        """
        encoded = self.geo_rd_encoder(group_weights)
        reconstructed = self.geo_rd_decoder(encoded)
        geo_rd_loss = F.mse_loss(reconstructed, group_weights, reduction='none').mean(dim=1)
        return geo_rd_loss

    def _get_bundle_z_score(self, bundle_rd_loss):
        """计算 bundle RD loss 的 Z-score.

        bundle_rd_loss = feature_rd_loss + lambda_geo * geo_rd_loss
        联合考虑特征分布偏移和群结构偏移。
        """
        if not self.bundle_loss_record.length > 2:
            return torch.zeros_like(bundle_rd_loss)
        mean = self.bundle_loss_record.mean
        stddev = self.bundle_loss_record.stddev
        z_score = (bundle_rd_loss - mean) / stddev
        z_score = torch.abs(z_score)
        return z_score

    def _add_bundle_record(self, bundle_rd_loss):
        """将 bundle RD loss 加入运行统计。"""
        self.bundle_loss_record.add_record(bundle_rd_loss.detach().cpu())


class SEMAModules(nn.Module):
    """一层 ViT 中所有适配器的管理器 (Bundle-aware 版本)。

    核心功能:
    1. 维护该层的适配器列表 (self.adapters)
    2. Bundle Router (self.router): 结合特征 + 群权重进行 softmax 路由
       - Fiber Router 部分: x.mean(dim=1) → 任务纤维选择
       - Group Router 联合: concat(x.mean(dim=1), group_weights) → 结构群 + 任务纤维联合选择
    3. 自扩展检测: 基于 bundle_z_score 判断是否添加新适配器
    4. 冻结旧适配器: 训练后锁定，保证不遗忘

    路由机制:
        router_input = [x_mean | group_weights]  (Bundle Router, 如果 use_bundle_router=True)
        router_input = x_mean                    (Fiber Router, 原始行为)
        w = softmax(router(router_input))
        output = MLP(x) + Σ w_i · Adapter_i(x)

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

        # Bundle Router: 结合群权重和特征进行路由
        # router_input_dim = D       (原始 Fiber Router)
        # router_input_dim = D + K   (Bundle Router: 结构群 G × 纤维 F 联合)
        self.use_bundle_router = (
            getattr(config, 'use_bundle_router', True)
            and getattr(config, 'use_group_pos', False)
        )
        self.router_input_dim = config.d_model
        if self.use_bundle_router:
            self.router_input_dim += getattr(config, 'num_groups', 4)
        self.router = nn.Linear(self.router_input_dim, 1)
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
        Bundle Router 版本使用 router_input_dim (D+K 或 D)。
        """
        self.new_router = nn.Linear(self.router_input_dim, 1).to(self._device())

    def fix_router(self):
        """将新路由器列合并到路由器中。

        训练完成后调用:
        1. 将旧路由器权重和新路由器权重拼接
        2. 旧列保持不变（已冻结），新列被纳入
        3. 新的合并后的路由器替换旧路由器
        Bundle Router 版本使用 router_input_dim。
        """
        trained_router = nn.Linear(
            self.router_input_dim, len(self.adapters)
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

    def forward(self, x, group_info=None):
        """Bundle-aware 前向传播: 路由 + 加权组合 + 扩展检测。

        Bundle Router: 如果 group_info 存在且 use_bundle_router=True,
        router 的输入从 x.mean(dim=1) 扩展为 [x.mean(dim=1) | group_weights],
        实现结构群 G 与任务纤维 F 的联合选择。

        扩展检测: 基于 bundle_z_score (特征+群权重的联合重建误差 Z-score)
        判断是否触发扩展, 而非仅基于特征 Z-score。

        Args:
            x: [B, N, d_model] ViT 块输入
            group_info: 来自 GroupRoutedPositionalEncoding 的字典, 或 None

        Returns:
            dict: {
                func_out, rd_loss, geo_rd_loss, bundle_rd_loss,
                z_score, bundle_z_score, added, router_weights
            }
        """
        zero = torch.tensor(0., device=x.device)
        rd_loss = zero
        geo_rd_loss = zero
        bundle_rd_loss = zero
        added = False

        # 浅层不进行扩展检测
        not_addition_layer = (
            self.layer_id < self.adapt_start_layer
            or self.layer_id > self.adapt_end_layer
        )

        if not_addition_layer:
            # 浅层: 直接用最后一个适配器, 无路由, 无 RD
            func_out, _, geo_rd_i, _, _, _ = self.adapters[-1](x, group_info=group_info)
            mask = None
            z_score_avg = zero
            bundle_z_avg = zero
        else:
            # 所有适配器前向传播 (bundle-aware)
            func_outs, rd_losses, geo_rd_losses = [], [], []
            bundle_rd_losses, z_scores, bundle_z_scores = [], [], []

            for adapter in self.adapters:
                f_out, rd_i, geo_i, b_rd_i, z_i, bz_i = adapter(x, group_info=group_info)
                func_outs.append(f_out)
                rd_losses.append(rd_i)
                geo_rd_losses.append(geo_i)
                bundle_rd_losses.append(b_rd_i)
                z_scores.append(z_i)
                bundle_z_scores.append(bz_i)

            func_outs = torch.stack(func_outs)           # [M, B, N, d]
            rd_losses = torch.stack(rd_losses)           # [M, B]
            geo_rd_losses = torch.stack(geo_rd_losses)   # [M, B]
            bundle_rd_losses = torch.stack(bundle_rd_losses)  # [M, B]
            z_scores = torch.stack(z_scores)             # [M, B]
            bundle_z_scores = torch.stack(bundle_z_scores)    # [M, B]

            # ---- 扩展检测 (Bundle-aware) ----
            # 使用 bundle_z_score (联合特征+群权重偏移) 进行检测
            # 若无 group_info, 退化为原始 z_score 检测
            detect_scores = bundle_z_scores if (group_info is not None) else z_scores
            z_score_avg = detect_scores.mean()

            addition_criteria = (
                detect_scores.mean(dim=1).min() > self.config.exp_threshold
                and self.layer_id >= self.adapt_start_layer
                and self.layer_id <= self.adapt_end_layer
                and not self.added_for_task
                and self.detecting_outlier
            )

            if addition_criteria:
                # 触发扩展!
                self.add_adapter()
                out = {
                    "func_out": torch.zeros_like(func_outs[0]),
                    "rd_loss": zero,
                    "geo_rd_loss": zero,
                    "bundle_rd_loss": zero,
                    "z_score": zero,
                    "bundle_z_score": zero,
                    "added": True,
                    "router_weights": None,
                }
                return out

            # ---- Bundle Router (结构群 G × 纤维 F 联合选择) ----
            if group_info is not None and self.use_bundle_router:
                router_input = torch.cat(
                    [x.mean(dim=1), group_info["group_weights"]], dim=-1
                )  # [B, D+K]
            else:
                router_input = x.mean(dim=1)  # [B, D]

            logits = self.router(router_input)  # [B, M]
            if self.new_router is not None:
                new_logits = self.new_router(router_input)  # [B, 1]
                logits = torch.cat([logits, new_logits], dim=1)  # [B, M+1]
            mask = torch.softmax(logits, dim=1)  # [B, M(+1)]

            # 加权求和: func_out = Σ mask_i * func_out_i
            func_out = (
                func_outs * mask.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
            ).sum(dim=0)

            # rd_loss: 如果有新适配器, 用最新 RD 的损失; 否则为 0
            if self.adapters[-1].newly_added:
                rd_loss = rd_losses[-1].mean()
                geo_rd_loss = geo_rd_losses[-1].mean()
                bundle_rd_loss = bundle_rd_losses[-1].mean()
            else:
                rd_loss = zero
                geo_rd_loss = geo_rd_losses[-1].mean() if group_info is not None else zero
                bundle_rd_loss = zero

            bundle_z_avg = bundle_z_scores.mean()

        out = {
            "func_out": func_out,
            "rd_loss": rd_loss,
            "geo_rd_loss": geo_rd_loss,
            "bundle_rd_loss": bundle_rd_loss,
            "z_score": z_score_avg,
            "bundle_z_score": bundle_z_avg,
            "added": added,
            "router_weights": mask,
        }
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
        """冻结所有表征描述器 (含 geometry RD)。

        冻结后 RD 不再更新统计和梯度，
        作为固定的分布检测器保留。
        同时冻结 geometry_rd 和 bundle_loss_record。
        """
        adapter_ls = self.adapters
        for adapter in adapter_ls:
            if adapter.rd is not None:
                for param in adapter.rd.parameters():
                    param.requires_grad = False
                    param._grad = None
                adapter.rd_loss_record.updating = False
            # 冻结 geometry_rd
            if adapter.geo_rd_encoder is not None:
                for param in adapter.geo_rd_encoder.parameters():
                    param.requires_grad = False
                    param._grad = None
                for param in adapter.geo_rd_decoder.parameters():
                    param.requires_grad = False
                    param._grad = None
                adapter.bundle_loss_record.updating = False
