"""
李群 SEMA 模块 — 基于测地线距离的自扩展检测
============================================

这是 SEMA 的 Lie Group 变体核心模块。与标准 SEMA (sema_block.py) 的区别:

  标准 SEMA:
    - Adapter 使用无约束的瓶颈 MLP
    - 扩展检测: 自编码器 (AE) 重建误差 + Z-score
    - Router 组合

  Lie-SEMA:
    - Adapter 使用 Stiefel 约束的 LieAdapter (down_proj ∈ St(768,16))
    - 扩展检测: Stiefel 流形上的测地线距离
    - Router 不变 (仍是 softmax 加权组合)

数学原理:
  Stiefel 流形上的测地线距离度量了不同 Adapter 学习的"视角"的差异:
    d_geo(W_i, W_new) > τ  → 新任务与所有旧 Adapter 差异显著 → 触发扩展

  这替代了 SEMA 的 AE+Z-score 机制, 提供有黎曼度量的几何判断。
"""

import torch
import torch.nn as nn
from typing import List
import copy
import logging

from backbones.lie_adapter import LieAdapter
from backbones.sema_components import AE, Records


class LieAdapterModule(nn.Module):
    """李群约束的适配器单元 (Lie Group Constrained Adapter Module)。

    包含:
      - functional: LieAdapter (down_proj ∈ Stiefel, up_proj free)
      - rd: AE (表征描述器, 保留用于 rd 阶段训练)
      - rd_loss_record: Records (运行统计, 保留用于 rd loss 追踪)
    """

    def __init__(self, config, adapter_id, writer):
        super().__init__()
        self.config = config
        self.functional = LieAdapter(
            config, adapter_id=adapter_id,
            dropout=0.1,
            adapter_scalar=config.ffn_adapter_scalar,
            adapter_layernorm_option=getattr(config, 'ffn_adapter_layernorm_option', 'none'),
        )
        layer_id = int(adapter_id.split('.')[0])
        self.not_addition_layer = (
            layer_id < config.adapt_start_layer
            or layer_id > config.adapt_end_layer
        )
        self.rd = None if self.not_addition_layer else AE(self.config)
        self.newly_added = True
        self.adapter_id = adapter_id
        self.writer = writer
        self.rd_loss_record = Records(max_len=config.buffer_size)

    def forward(self, x):
        """前向传播: 功能输出 + 可选的 rd_loss + z_score。

        z_score 保留用于兼容, 但 Lie-SEMA 的扩展检测使用测地线距离而非 z_score。
        """
        func_out = self.functional(x)
        if self.not_addition_layer or self.rd is None:
            rd_loss = torch.tensor(0., device=x.device)
            return func_out, rd_loss, torch.zeros_like(rd_loss)
        else:
            rd_loss = self.rd.compute_reconstruction_loss(x)
        z_score = self.get_z_score_deviation(rd_loss)
        if self.training:
            self.add_z_score_record(rd_loss)
        return func_out, rd_loss, z_score

    def get_z_score_deviation(self, rd_loss):
        """保留 Z-score 计算 (兼容 rd 阶段训练跟踪)。"""
        mean, stddev = self.rd_loss_record.mean, self.rd_loss_record.stddev
        if not self.rd_loss_record.length > 2:
            return torch.zeros_like(rd_loss)
        z_score = (rd_loss - mean) / stddev
        return torch.abs(z_score)

    def add_z_score_record(self, rd_loss):
        self.rd_loss_record.add_record(rd_loss.detach().cpu())

    def project_(self):
        """将 functional.down_proj 投影到 Stiefel 流形。"""
        self.functional.project_()


class LieSEMAModules(nn.Module):
    """李群 SEMA 模块管理器 — 基于测地线距离的自扩展。

    每层 ViT Block 中的一个 LieSEMAModules 管理该层的所有 LieAdapter,
    负责:
      1. 维护适配器列表
      2. 软路由器 (加权组合, 与标准 SEMA 相同)
      3. 自扩展检测 (改进: 测地线距离替代 Z-score)
      4. 冻结逻辑

    扩展检测逻辑:
      - 前向时, 对每个 batch, 检查是否所有旧 Adapter 与新数据的
        "匹配度"不足 (测地线距离判断)
      - 由于测地线距离不依赖 batch 计算 (它是参数空间的距离),
        我们在 _train_new 完成后检查, 而非在 forward 中
    """

    def __init__(self, config, layer_id, writer):
        super().__init__()
        self.adapters: List[LieAdapterModule] = nn.ModuleList()
        self.config = config
        self.layer_id = layer_id
        self.writer = writer
        self.adapt_start_layer = config.adapt_start_layer
        self.adapt_end_layer = config.adapt_end_layer
        self.geo_threshold = getattr(config, 'geo_threshold', 0.5)  # 测地线距离阈值

        # 初始一个 Adapter
        self.add_adapter(initialize=True)
        self.added_adapter = 0

        # Router (不变)
        self.router = nn.Linear(config.d_model, 1)
        self.new_router = None
        self.detecting_outlier = False
        self.added_for_task = True
        self.newly_added = True

    @property
    def num_adapters(self):
        return len(self.adapters)

    def _device(self):
        if hasattr(self, 'router') and self.router is not None:
            return self.router.weight.device
        if len(self.adapters) > 0:
            return next(self.adapters[0].parameters()).device
        return torch.device('cpu')

    def set_new_router(self):
        self.new_router = nn.Linear(self.config.d_model, 1).to(self._device())

    def fix_router(self):
        """合并新路由列到主路由器。"""
        trained_router = nn.Linear(self.config.d_model, len(self.adapters)).to(self._device())
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
        adapter_id = f"{self.layer_id}.{len(self.adapters)}"
        new_adapter = LieAdapterModule(self.config, adapter_id, self.writer).to(self._device())
        self.newly_added = True
        self.added_for_task = True
        self.adapters.append(new_adapter)
        if not initialize:
            self.set_new_router()
        logging.info(f"LieAdapter {adapter_id} added at block {self.layer_id}")

    # ────── 测地线距离扩展检测 ──────

    def check_geodesic_expansion(self):
        """基于测地线距离判断是否需要扩展。

        检查最新的 Adapter 与所有旧 Adapter 之间的最小测地线距离。
        如果都超过阈值, 说明新任务学习到的几何基与历史差异过大,
        需要为新任务添加独立的 Adapter。

        调用时机: _train_new() 完成后 (此时新 Adapter 已有训练后的权重)。

        Returns:
            (should_expand, min_distance): bool 和 float 距离值
        """
        if len(self.adapters) < 2:
            return False, 0.0

        is_detect_layer = (
            self.layer_id >= self.adapt_start_layer
            and self.layer_id <= self.adapt_end_layer
        )
        if not is_detect_layer:
            return False, 0.0

        newest = self.adapters[-1]           # 刚训练完的 Adapter
        distances = []
        for i in range(len(self.adapters) - 1):  # 与所有旧 Adapter 比较
            dist = newest.functional.geodesic_distance_to(
                self.adapters[i].functional
            )
            distances.append(dist)

        min_dist = min(distances) if distances else 0.0
        should_expand = min_dist > self.geo_threshold

        if should_expand:
            logging.info(
                f"Block {self.layer_id}: geodesic min_dist={min_dist:.4f} > "
                f"threshold={self.geo_threshold} → expansion triggered"
            )

        return should_expand, float(min_dist)

    # ────── 前向传播 ──────

    def forward(self, x):
        rd_loss = torch.tensor(0., device=x.device)
        added = False

        is_detect_layer = (
            self.layer_id >= self.adapt_start_layer
            and self.layer_id <= self.adapt_end_layer
        )

        if not is_detect_layer:
            func_out, _, _ = self.adapters[-1](x)
        else:
            func_outs, rd_losses, _ = [], [], []
            for adapter in self.adapters:
                func_out_i, rd_loss_i, _ = adapter(x)
                func_outs.append(func_out_i)
                rd_losses.append(rd_loss_i)

            func_outs = torch.stack(func_outs)
            rd_losses = torch.stack(rd_losses)

            # Router 加权组合
            logits = self.router(x.mean(dim=1))
            if self.new_router is not None:
                new_logits = self.new_router(x.mean(dim=1))
                logits = torch.cat([logits, new_logits], dim=1)
            mask = torch.softmax(logits, dim=1)
            func_out = (func_outs * mask.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)).sum(dim=0)

            if self.adapters[-1].newly_added:
                rd_loss = rd_losses[-1].mean()
            else:
                rd_loss = torch.tensor(0., device=x.device)

        out = {"func_out": func_out, "rd_loss": rd_loss, "added": added}
        return out

    # ────── 任务结束处理 ──────

    def end_of_task_training(self):
        self.freeze_functional()
        self.freeze_rd()
        self.reset_newly_added_status()
        self.added_for_task = False

    def reset_newly_added_status(self):
        self.newly_added = False
        for adapter in self.adapters:
            adapter.newly_added = False

    def freeze_functional(self):
        for adapter in self.adapters:
            for param in adapter.functional.parameters():
                param.requires_grad = False
                param._grad = None
        if self.new_router is not None:
            self.fix_router()
        for param in self.router.parameters():
            param.requires_grad = False
            param._grad = None

    def freeze_rd(self):
        for adapter in self.adapters:
            if adapter.rd is not None:
                for param in adapter.rd.parameters():
                    param.requires_grad = False
                    param._grad = None
                adapter.rd_loss_record.updating = False

    # ────── Stiefel 投影 (全局操作, 不计入单个模块) ──────

    def project_all_(self):
        """将本层所有 Adapter 的 down_proj 投影到 Stiefel 流形。

        在 optimizer.step() 后调用, 对每个可训练的 Adapter 执行 SVD 投影。
        """
        for adapter in self.adapters:
            adapter.project_()
