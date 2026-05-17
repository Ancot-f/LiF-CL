"""
Flat MoE Adapter — 扁平混合专家适配器
======================================

设计思路:
  每个专家 = 独立瓶颈 MLP, 不做群绑定。
  群几何结构 (SO 正交 / LR 低秩) 降级为可选正则化。
  路由器扁平化: 直接输出 E 个专家权重, 无层次结构。

与 Group-MoE 的关键区别:
  - 专家 = MLP(down→ReLU→up), 不是群操作 (SO / LR / Affine)
  - 无 MambaFlow (去掉 selective_scan Python 循环)
  - 扁平路由: 1 层 softmax 选专家, 而非 GroupRouter→ExpertRouter
  - 每专家独立 RD: E 个 AE, 而非 G 个群 AE
  - 群约束可选: so_reg / lr_reg 作为权重正则, 不强绑定

数据流:
  z = LN(x).mean(dim=1)              — 池化特征
  w = softmax(FlatRouter(z))          — 专家权重 [B, E]
  a = Σ_e w_e * Expert_e(x_norm)     — 加权专家输出 [B, N, D]
  out = MLP(x) + a                    — 残差 (MLP 冻结不参与)

扩展检测 (继承原 SEMA):
  每专家维护 AE + RunningRecords
  z_score_e = |MSE(AE_e(z), z) - mean_e| / std_e
  if max(z_score) > threshold → 添加新专家

组件:
  - ExpertAdapter:    瓶颈 MLP 专家
  - ExpertBank:       专家池 (可扩展)
  - FlatRouter:       扁平路由器
  - ExpertAwareAE:    逐专家自编码器
  - RunningRecords:   运行统计 (逐专家)
  - ExpertMoEAdapter: 完整适配器
  - ExpertMoEModules: 层级管理器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import math
import logging


# ═══════════════════════════════════════════════════════════════════════════════
# ExpertAdapter — 瓶颈 MLP 专家
# ═══════════════════════════════════════════════════════════════════════════════

class ExpertAdapter(nn.Module):
    """瓶颈 MLP 专家: down → ReLU → up.

    每个专家是一个独立的小型 adapter, 参数独立于其他专家。
    初始化策略与原 SEMA Adapter 一致:
      - down_proj: Kaiming Uniform
      - up_proj: 全零 (适配器从零开始, 不干扰预训练)

    Args:
        dim:         输入/输出维度 (768)
        bottleneck:  瓶颈维度 (16)
    """

    def __init__(self, dim, bottleneck=16):
        super().__init__()
        self.dim = dim
        self.bottleneck = bottleneck
        self.down_proj = nn.Linear(dim, bottleneck)
        self.up_proj = nn.Linear(bottleneck, dim)
        self.act = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        """LoRA 风格初始化: down Kaiming, up 全零。"""
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)       # 零初始化 - 关键
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        """x: [B, N, D] → [B, N, D]"""
        return self.up_proj(self.act(self.down_proj(x)))

    def so_regularization(self):
        """可选 SO 正则化: ||down_proj_small^T down_proj_small - I||^2。
        取 down_proj 的前 bottleneck 列, 约束其近似正交。
        """
        W = self.down_proj.weight  # [bottleneck, dim]
        WTW = W @ W.T              # [bottleneck, bottleneck]
        eye = torch.eye(self.bottleneck, device=W.device)
        return torch.norm(WTW - eye, p='fro') ** 2

    def lr_regularization(self):
        """可选低秩正则化: nuclear norm of down_proj (鼓励低秩)。"""
        return torch.linalg.norm(self.down_proj.weight, ord='nuc')


# ═══════════════════════════════════════════════════════════════════════════════
# ExpertBank — 专家池
# ═══════════════════════════════════════════════════════════════════════════════

class ExpertBank(nn.Module):
    """可扩展专家池: Expert_0, Expert_1, ...

    每个专家是独立的 ExpertAdapter (瓶颈 MLP)。
    支持 add_expert() 在运行时扩展。

    Args:
        dim:         特征维度
        bottleneck:  瓶颈维度
        init_experts: 初始专家数量 (默认 1)
    """

    def __init__(self, dim, bottleneck=16, init_experts=1):
        super().__init__()
        self.dim = dim
        self.bottleneck = bottleneck
        self.experts = nn.ModuleList([
            ExpertAdapter(dim, bottleneck) for _ in range(init_experts)
        ])

    def add_expert(self):
        """添加新专家并返回其索引。自动迁移到已有专家所在设备。"""
        new_expert = ExpertAdapter(self.dim, self.bottleneck)
        if len(self.experts) > 0:
            target_device = next(self.experts[0].parameters()).device
            new_expert = new_expert.to(target_device)
        self.experts.append(new_expert)
        logging.info(f"ExpertBank: added expert #{len(self.experts) - 1}")
        return len(self.experts) - 1

    def num_experts(self):
        return len(self.experts)

    def forward(self, x, weights=None):
        """加权组合所有专家输出。

        Args:
            x:       [B, N, D] 输入
            weights: [B, E] 专家权重 (None 表示只用最后一个)

        Returns:
            out:     [B, N, D] 加权输出
        """
        if weights is None:
            return self.experts[-1](x)

        outputs = [expert(x) for expert in self.experts]  # E × [B, N, D]
        stacked = torch.stack(outputs, dim=0)              # [E, B, N, D]
        w = weights.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # [E, B, 1, 1]
        return (stacked * w).sum(dim=0)                    # [B, N, D]

    def so_regularization(self):
        """所有专家的 SO 正则化之和。"""
        return sum(e.so_regularization() for e in self.experts)

    def lr_regularization(self):
        """所有专家的低秩正则化之和。"""
        return sum(e.lr_regularization() for e in self.experts)


# ═══════════════════════════════════════════════════════════════════════════════
# FlatRouter — 扁平路由器
# ═══════════════════════════════════════════════════════════════════════════════

class FlatRouter(nn.Module):
    """扁平路由器: 直接从池化特征 → E 个专家权重。

    输入: concat(cls_token, mean(tokens), std(tokens), opt_z_scores)
    输出: softmax(logits / tau) → [B, E]

    z-score 校正 (可选): logit_e = MLP(h)_e - beta * stopgrad(z_e)
      高异常 → 降低该专家权重, 让其他专家有机会处理

    Args:
        dim:        特征维度
        init_experts: 初始专家数
        beta:       z-score 校正强度
        tau:        softmax 温度
    """

    def __init__(self, dim, init_experts=1, beta=0.1, tau=1.0,
                 max_experts=32):
        super().__init__()
        self.dim = dim
        self.num_experts = init_experts
        self.beta = beta
        self.tau = tau

        # 路由输入: cls(D) + mean(D) + std(D) + z_scores(max_E)
        # 预分配最大专家数的输入维度, 避免扩展时维度变化
        self.max_experts = max_experts
        router_input_dim = dim * 3 + max_experts
        router_hidden = max(dim // 2, 64)

        self.router = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, init_experts),
        )
        self._init_weights()

    def _init_weights(self):
        """稳定初始化: 最后一层小权重 + 零 bias, 初始路由接近均匀。"""
        last = self.router[-1]
        nn.init.trunc_normal_(last.weight, std=0.02)
        nn.init.zeros_(last.bias)

    def expand(self, new_num_experts):
        """扩展路由器输出维度 (新增专家列)。

        旧列权重保留并冻结 (梯度清零), 新列随机初始化可训练。
        输入维度预分配为 max_experts, 不随扩展改变。
        """
        old_num = self.num_experts
        if new_num_experts <= old_num or new_num_experts > self.max_experts:
            return

        old_output = self.router[-1]  # nn.Linear(hidden, old_num)
        new_output = nn.Linear(
            old_output.in_features, new_num_experts,
            device=old_output.weight.device,
        )
        nn.init.trunc_normal_(new_output.weight, std=0.02)
        nn.init.zeros_(new_output.bias)
        with torch.no_grad():
            new_output.weight.data[:old_num] = old_output.weight.data
            new_output.bias.data[:old_num] = old_output.bias.data

        # 旧列梯度清零 hook: 只训练新列
        new_output.weight.requires_grad_(True)
        new_output.bias.requires_grad_(True)

        def _zero_old_grad(grad):
            grad[:old_num] = 0
            return grad
        new_output.weight.register_hook(_zero_old_grad)
        new_output.bias.register_hook(_zero_old_grad)

        self.router[-1] = new_output
        self.num_experts = new_num_experts
        logging.info(f"FlatRouter: {old_num} -> {new_num_experts} experts")

    def forward(self, x, z_scores=None):
        """扁平路由前向传播。

        Args:
            x:        [B, N, D] token 序列
            z_scores: [B, E] 逐专家 z-score (可选)

        Returns:
            expert_weights: [B, E] softmax 权重
        """
        B, N, D = x.shape
        cls_token = x[:, 0]       # [B, D]
        mean_tok = x.mean(dim=1)  # [B, D]
        std_tok = x.std(dim=1)    # [B, D]

        # z_scores 填充到 max_experts (不足部分填零)
        if z_scores is not None:
            zs = z_scores
        else:
            zs = torch.zeros(B, self.num_experts, device=x.device)
        # 不足 max_experts 部分填零
        if zs.shape[1] < self.max_experts:
            zs = torch.cat([zs, torch.zeros(B, self.max_experts - zs.shape[1],
                                            device=x.device)], dim=-1)
        parts = [cls_token, mean_tok, std_tok, zs]

        router_input = torch.cat(parts, dim=-1)  # [B, 3D+E]
        logits = self.router(router_input)        # [B, E]

        # z-score 校正: 高异常 → 降低该专家权重
        if z_scores is not None:
            logits = logits - self.beta * z_scores.detach()

        return F.softmax(logits / self.tau, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# ExpertAwareAE — 逐专家自编码器 (RD)
# ═══════════════════════════════════════════════════════════════════════════════

class ExpertAwareAE(nn.Module):
    """逐专家自编码器: 每个专家有独立的 encoder→decoder。

    L_RD = Σ_e w_e * MSE(AE_e(z), z)

    用于扩展检测:
      - 重建误差小 → 该专家能解释当前样本
      - 重建误差大 → 分布偏移 → 触发扩展

    Args:
        dim:        特征维度 (768)
        rd_dim:     AE 压缩维度
        init_experts: 初始专家数
    """

    def __init__(self, dim, rd_dim=None, init_experts=1):
        super().__init__()
        self.dim = dim
        self.rd_dim = rd_dim or max(dim // 6, 32)
        self.num_experts = init_experts

        self.encoders = nn.ModuleList([
            nn.Linear(dim, self.rd_dim) for _ in range(init_experts)
        ])
        self.decoders = nn.ModuleList([
            nn.Linear(self.rd_dim, dim) for _ in range(init_experts)
        ])
        self._init_weights()

    def _init_weights(self):
        for enc, dec in zip(self.encoders, self.decoders):
            nn.init.kaiming_uniform_(enc.weight, a=math.sqrt(5))
            nn.init.zeros_(enc.bias)
            nn.init.kaiming_uniform_(dec.weight, a=math.sqrt(5))
            nn.init.zeros_(dec.bias)

    def add_expert_ae(self):
        """为新专家添加 encoder+decoder。"""
        new_enc = nn.Linear(self.dim, self.rd_dim)
        new_dec = nn.Linear(self.rd_dim, self.dim)
        nn.init.kaiming_uniform_(new_enc.weight, a=math.sqrt(5))
        nn.init.zeros_(new_enc.bias)
        nn.init.kaiming_uniform_(new_dec.weight, a=math.sqrt(5))
        nn.init.zeros_(new_dec.bias)

        if len(self.encoders) > 0:
            target_device = next(self.encoders[0].parameters()).device
            new_enc = new_enc.to(target_device)
            new_dec = new_dec.to(target_device)

        self.encoders.append(new_enc)
        self.decoders.append(new_dec)
        self.num_experts += 1

    def forward_all(self, z):
        """所有专家编码→解码。

        Args:
            z: [B, dim] 池化特征

        Returns:
            reconstructions: [E, B, dim]
        """
        out = []
        for e in range(self.num_experts):
            out.append(self.decoders[e](self.encoders[e](z)))
        return torch.stack(out, dim=0)  # [E, B, dim]

    def compute_per_expert_rd(self, z, expert_weights):
        """计算逐专家加权 RD 损失。

        Args:
            z:              [B, dim]
            expert_weights: [B, E]

        Returns:
            rd_loss:        [B] 加权 RD 损失
            per_expert_loss: [B, E] 逐专家损失
        """
        B, D = z.shape
        E = self.num_experts
        all_rec = self.forward_all(z)  # [E, B, D]

        per_expert_loss = torch.zeros(B, E, device=z.device)
        for e in range(E):
            per_expert_loss[:, e] = F.mse_loss(
                all_rec[e], z, reduction='none'
            ).mean(dim=-1)

        rd_loss = (per_expert_loss * expert_weights).sum(dim=-1)
        return rd_loss, per_expert_loss


# ═══════════════════════════════════════════════════════════════════════════════
# RunningRecords — 运行统计缓冲区 (逐专家)
# ═══════════════════════════════════════════════════════════════════════════════

class RunningRecords:
    """在线运行统计: 维护每个专家 RD 损失的均值和标准差。

    Z-score = |当前损失 - 历史均值| / 历史标准差
    Z > threshold → 分布偏移 → 触发扩展

    Args:
        max_len: 缓冲区最大容量
    """

    def __init__(self, max_len=500):
        self._max_len = max_len
        self._curr_len = 0
        self.record = torch.zeros(max_len)
        self._mean = 0.0
        self._var = 0.0
        self.updating = True

    @property
    def length(self):
        return self._curr_len

    @property
    def mean(self):
        return self._mean

    @property
    def stddev(self):
        return math.sqrt(max(self._var, 1e-8))

    def add_record(self, v):
        if not self.updating:
            return
        v = v.detach().cpu()
        if self._curr_len < self._max_len:
            place_left = self._max_len - self._curr_len
            if place_left > len(v):
                self.record[self._curr_len:self._curr_len + len(v)] = v
                self._curr_len += len(v)
            else:
                self.record[self._curr_len:] = v[:place_left]
                self._curr_len = self._max_len
        else:
            self.record = torch.cat([self.record, v])
            self.record = self.record[len(v):]

        self._mean = float(torch.mean(self.record[:self._curr_len]))
        self._var = float(torch.var(self.record[:self._curr_len]))


# ═══════════════════════════════════════════════════════════════════════════════
# ExpertMoEAdapter — 完整 Flat MoE 适配器
# ═══════════════════════════════════════════════════════════════════════════════

class ExpertMoEAdapter(nn.Module):
    """Flat MoE 适配器: ExpertBank + FlatRouter + ExpertAwareAE。

    数据流:
      1. z_pooled = LN(x).mean(dim=1)          — 池化特征
      2. w = FlatRouter(z_pooled)               — 专家权重 [B, E]
      3. a = Σ_e w_e * Expert_e(LN(x))         — 加权专家输出
      4. RD: L_rd = Σ_e w_e * MSE(AE_e(z), z)  — 重建损失

    Args:
        config:   全局配置
        layer_id: ViT 层索引
    """

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        d_model = config.d_model
        bottleneck = getattr(config, 'ffn_num', 16)
        init_experts = getattr(config, 'init_experts', 1)

        # 共享 LayerNorm
        self.norm = nn.LayerNorm(d_model)

        # ExpertBank
        self.expert_bank = ExpertBank(d_model, bottleneck, init_experts)

        # FlatRouter
        self.router = FlatRouter(
            d_model, init_experts=init_experts,
            beta=getattr(config, 'router_beta', 0.1),
            tau=getattr(config, 'router_tau', 1.0),
        )

        # 残差门控 (零初始化)
        self.gamma = nn.Parameter(torch.zeros(1))

        # ExpertAwareAE + RunningRecords (仅深层)
        self.not_addition_layer = (
            layer_id < config.adapt_start_layer
            or layer_id > config.adapt_end_layer
        )
        if not self.not_addition_layer:
            self.expert_ae = ExpertAwareAE(
                d_model,
                rd_dim=getattr(config, 'rd_dim', 64),
                init_experts=init_experts,
            )
            self.per_expert_records = [
                RunningRecords(max_len=getattr(config, 'buffer_size', 500))
                for _ in range(init_experts)
            ]
            # 群约束标志
            self.use_so_reg = getattr(config, 'use_so_reg', True)
            self.use_lr_reg = getattr(config, 'use_lr_reg', False)
        else:
            self.expert_ae = None
            self.per_expert_records = None
            self.use_so_reg = False
            self.use_lr_reg = False

        self.newly_added = True

    def _compute_z_scores(self, per_expert_loss, detach=True):
        """从逐专家 RD 损失计算 z-score。"""
        B, E = per_expert_loss.shape
        z_scores = torch.zeros(B, E, device=per_expert_loss.device)
        for e in range(E):
            rec = self.per_expert_records[e]
            if rec.length > 2:
                loss_e = per_expert_loss[:, e].detach() if detach else per_expert_loss[:, e]
                z_scores[:, e] = torch.abs((loss_e - rec.mean) / rec.stddev)
        return z_scores

    def forward(self, x, compute_rd=True):
        """Flat MoE 前向传播。

        Args:
            x:          [B, N, d_model] ViT 块输出
            compute_rd: 是否计算 RD (func 阶段跳过)

        Returns:
            dict: func_out, rd_loss, z_scores, expert_weights, added
        """
        B, N, D = x.shape
        x_norm = self.norm(x)

        # ── 路由 ──
        use_ae = (compute_rd and not self.not_addition_layer
                  and self.expert_ae is not None)
        if use_ae:
            z_pooled = x_norm.mean(dim=1)
            # 用均匀权重获取无偏 RD
            uniform_w = torch.ones(B, self.expert_bank.num_experts(),
                                   device=x.device) / self.expert_bank.num_experts()
            _, per_exp_rd = self.expert_ae.compute_per_expert_rd(z_pooled, uniform_w)
            z_scores = self._compute_z_scores(per_exp_rd)
            expert_weights = self.router(x_norm, z_scores=z_scores)
        else:
            expert_weights = self.router(x_norm, z_scores=None)

        # ── 专家混合 ──
        a = self.expert_bank(x_norm, weights=expert_weights)  # [B, N, D]

        # ── RD 损失 (仅 compute_rd) ──
        rd_loss = torch.tensor(0.0, device=x.device)
        z_scores_out = torch.zeros(B, self.expert_bank.num_experts(), device=x.device)

        if use_ae:
            z_pooled = x_norm.mean(dim=1)
            rd_loss, per_exp_rd = self.expert_ae.compute_per_expert_rd(
                z_pooled, expert_weights
            )
            rd_loss = rd_loss.mean()
            if self.training:
                for e in range(self.expert_bank.num_experts()):
                    self.per_expert_records[e].add_record(per_exp_rd[:, e])
            z_scores_out = self._compute_z_scores(per_exp_rd, detach=False)

        return {
            "func_out": a,
            "rd_loss": rd_loss,
            "z_scores": z_scores_out,
            "expert_weights": expert_weights,
            "added": False,
        }

    def add_expert(self):
        """添加新专家: ExpertBank + Router + AE 全部扩展。"""
        new_idx = self.expert_bank.add_expert()
        self.router.expand(self.expert_bank.num_experts())
        if self.expert_ae is not None:
            self.expert_ae.add_expert_ae()
            self.per_expert_records.append(
                RunningRecords(max_len=getattr(self.config, 'buffer_size', 500))
            )
        return new_idx

    def orthogonality_error(self):
        """SO 正则化: 所有专家 down_proj 的正交误差之和。"""
        if self.use_so_reg:
            return self.expert_bank.so_regularization()
        return torch.tensor(0.0)

    def low_rank_error(self):
        """低秩正则化: 所有专家 down_proj 的核范数之和。"""
        if self.use_lr_reg:
            return self.expert_bank.lr_regularization()
        return torch.tensor(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# ExpertMoEModules — 层级管理器
# ═══════════════════════════════════════════════════════════════════════════════

class ExpertMoEModules(nn.Module):
    """层级 Flat MoE 管理器。

    管理:
      - 专家池 (ExpertBank)
      - 扩展检测 (z-score + 持续性)
      - 任务管理 (freeze, end_of_task)

    Args:
        config:   全局配置
        layer_id: ViT 层索引
    """

    def __init__(self, config, layer_id, writer=None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.writer = writer
        self.adapt_start_layer = config.adapt_start_layer
        self.adapt_end_layer = config.adapt_end_layer

        # ── 扩展检测状态 ──
        self.detecting_outlier = False
        self.added_for_task = False
        self.newly_added = True

        # ── 多 batch 持续性检测 ──
        expansion_patience = getattr(config, 'expansion_patience', 3)
        self._z_score_accum = torch.zeros(1)      # 动态大小
        self._z_score_count = 0
        self._expansion_patience = expansion_patience
        self._expansion_candidate = -1

        # ── 初始化适配器 ──
        self.adapters: List[ExpertMoEAdapter] = nn.ModuleList()
        self.add_adapter(initialize=True)

        self.expansion_count = 0

    @property
    def num_adapters(self):
        return len(self.adapters)

    def _device(self):
        if len(self.adapters) > 0:
            return next(self.adapters[0].parameters()).device
        return torch.device('cpu')

    def add_adapter(self, initialize=False):
        """添加新的 ExpertMoEAdapter。"""
        new_adapter = ExpertMoEAdapter(self.config, self.layer_id).to(self._device())
        self.newly_added = True
        self.added_for_task = True
        self.adapters.append(new_adapter)
        if not initialize:
            logging.info(f"ExpertMoEAdapter {self.layer_id} 已添加")

    def add_expert(self):
        """向当前适配器池中添加新专家。"""
        if self.adapters:
            adapter = self.adapters[-1]
            new_idx = adapter.add_expert()
            # 解冻新专家参数和 AE
            new_expert = adapter.expert_bank.experts[new_idx]
            for param in new_expert.parameters():
                param.requires_grad = True
            if adapter.expert_ae is not None:
                new_enc = adapter.expert_ae.encoders[new_idx]
                new_dec = adapter.expert_ae.decoders[new_idx]
                for param in new_enc.parameters():
                    param.requires_grad = True
                for param in new_dec.parameters():
                    param.requires_grad = True
            self.expansion_count += 1
            return new_idx
        return -1

    def forward(self, x, group_info=None):
        """Flat MoE 前向 + 扩展检测。

        Args:
            x: [B, N, D]

        Returns:
            dict: func_out, rd_loss, z_scores, expert_weights, added
        """
        zero = torch.tensor(0.0, device=x.device)
        not_addition_layer = (
            self.layer_id < self.adapt_start_layer
            or self.layer_id > self.adapt_end_layer
        )

        if not_addition_layer:
            adapter_out = self.adapters[-1](x, compute_rd=False)
            return {
                "func_out": adapter_out["func_out"],
                "rd_loss": zero,
                "z_scores": adapter_out["z_scores"],
                "expert_weights": adapter_out["expert_weights"],
                "added": False,
            }

        # 深层: compute_rd 仅在检测模式或 RD 训练时
        compute_rd = self.detecting_outlier or getattr(self, '_training_rd', False)
        adapter = self.adapters[-1]
        adapter_out = adapter(x, compute_rd=compute_rd)

        # ── 扩展检测 ──
        added = False
        if self.detecting_outlier and not self.added_for_task:
            z_scores = adapter_out.get("z_scores")
            if z_scores is not None and z_scores.shape[1] > 0:
                batch_z_mean = z_scores.mean(dim=0).detach().cpu()  # [E]

                # 确保累积器大小匹配
                E = batch_z_mean.shape[0]
                if self._z_score_accum.shape[0] != E:
                    new_accum = torch.zeros(E)
                    old_len = min(self._z_score_accum.shape[0], E)
                    new_accum[:old_len] = self._z_score_accum[:old_len]
                    self._z_score_accum = new_accum

                # 更新累积 z-score (运行均值)
                self._z_score_accum = (
                    self._z_score_accum * self._z_score_count
                    + batch_z_mean
                ) / (self._z_score_count + 1)
                self._z_score_count += 1

                best_idx = self._z_score_accum.argmax().item()
                max_z = self._z_score_accum[best_idx].item()

                if max_z > self.config.exp_threshold:
                    if (self._expansion_candidate == best_idx
                            and self._z_score_count >= self._expansion_patience):
                        self.add_expert()
                        self.added_for_task = True
                        added = True
                        logging.info(
                            f"Block {self.layer_id}: 添加专家 #{best_idx} "
                            f"(累积 z={max_z:.3f} > 阈值={self.config.exp_threshold}, "
                            f"持续 {self._z_score_count} batches)"
                        )
                        self._z_score_accum = torch.zeros(
                            self.adapters[-1].expert_bank.num_experts()
                        )
                        self._z_score_count = 0
                        self._expansion_candidate = -1
                    else:
                        self._expansion_candidate = best_idx
                else:
                    self._expansion_candidate = -1

        return {
            "func_out": adapter_out["func_out"],
            "rd_loss": adapter_out["rd_loss"],
            "z_scores": adapter_out["z_scores"],
            "expert_weights": adapter_out["expert_weights"],
            "added": added,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # 任务管理
    # ═══════════════════════════════════════════════════════════════════════

    def end_of_task_training(self):
        """冻结所有参数 + 停止 RD 统计更新 + 重置扩展状态。"""
        self.freeze_functional()
        self.freeze_rd()
        self.reset_newly_added_status()
        self.added_for_task = False
        self._z_score_accum.zero_()
        self._z_score_count = 0
        self._expansion_candidate = -1

    def reset_newly_added_status(self):
        self.newly_added = False
        for adapter in self.adapters:
            adapter.newly_added = False

    def freeze_functional(self):
        """冻结所有专家和路由器参数。"""
        for adapter in self.adapters:
            adapter.gamma.requires_grad_(False)
            for expert in adapter.expert_bank.experts:
                for param in expert.parameters():
                    param.requires_grad = False
            for param in adapter.router.parameters():
                param.requires_grad = False

    def freeze_rd(self):
        """冻结 AE 并停止统计更新。"""
        for adapter in self.adapters:
            if adapter.expert_ae is not None:
                for param in adapter.expert_ae.parameters():
                    param.requires_grad = False
                if adapter.per_expert_records:
                    for rec in adapter.per_expert_records:
                        rec.updating = False
