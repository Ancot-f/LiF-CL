"""
Group Basis MoE — 群基组合混合专家
====================================

核心理念:
  不把单个群操作当专家, 而是维护共享的"群基原子池",
  每个概念通过学到的基组合权重来描述。

  概念(z) = Σ_g w_g · ( Σ_k β_{g,k} · Basis_{g,k}(z) )
            ↑群权重       ↑群内基组合权重

与三种老方案的区别:
  - Option A (加权混合): 每专家固定 4 个标量, 表达力受限
  - Option B (序列组合): 固化组合顺序, O(k^N) 爆炸
  - Option C (扁平 MLP):  无几何结构, 失去可解释性
  - 群基组合:            有结构 + 可组合 + 几何可解释 + 语义距离可度量

扩展策略:
  Task 0: 初始化 G 个群 × K 个基的原子池, 训练组合路由器
  Task t: RD 检测到异常 → 每个群各加 1 个新基 (K → K+1)
          旧基冻结, 新基可训练, Router 输出维度随 K 增长

架构:
  GroupBasisBank:     G 个群 × K 个基 (可扩展)
  CompositionRouter:  输出 group_weights[B,G] + basis_weights[B,G,K]
  GroupAwareAE:       逐群 RD (G 个 AE)
  GroupBasisAdapter:  完整适配器
  GroupBasisModules:  层级管理器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import math
import logging


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Group Basis — 群基原子
# ═══════════════════════════════════════════════════════════════════════════════

class IdentityBasis(nn.Module):
    """恒等基: T(z) = 0 (零残差, 主路径已含 MLP)。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, z):
        return torch.zeros_like(z)


class SOBasis(nn.Module):
    """SO(r) 旋转基: z_SO = z @ R, R 被鼓励趋向 SO(r)。

    每个基是一个可学习的旋转矩阵。
    L_geo = ||R^T R - I||^2 作为软约束。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.R = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01)

    def forward(self, z):
        return z @ self.R

    def orthogonality_error(self):
        RTR = self.R.T @ self.R
        eye = torch.eye(self.dim, device=self.R.device)
        return torch.norm(RTR - eye, p='fro') ** 2


class LRBasis(nn.Module):
    """低秩基: z_LR = z + z @ A @ B, A∈R^{r×k}, B∈R^{k×r}。

    低秩修正仅改变少数方向的特征, 保持语义结构不变。
    """

    def __init__(self, dim, rank=None):
        super().__init__()
        self.dim = dim
        self.rank = rank or max(1, dim // 4)
        self.A = nn.Parameter(torch.randn(dim, self.rank) * 0.01 / math.sqrt(self.rank))
        self.B = nn.Parameter(torch.randn(self.rank, dim) * 0.01 / math.sqrt(self.rank))

    def forward(self, z):
        return z + z @ self.A @ self.B


class AffineBasis(nn.Module):
    """仿射基: z_Affine = z @ W + b。

    最灵活的基原子, 应谨慎使用。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.W = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        return z @ self.W + self.b


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GroupBasisBank — 群基原子池
# ═══════════════════════════════════════════════════════════════════════════════

class GroupBasisBank(nn.Module):
    """共享群基原子池: G 个群 × K 个基。

    群类型: Identity(1个), SO(K个), LR(K个), Affine(K个)
    总计: 1 + 3K 个基原子

    扩展: add_bases() → 每个可扩展群各加 1 个新基 (K → K+1)

    Args:
        dim:         瓶颈维度 r
        num_bases:   每群初始基数量 K
    """

    def __init__(self, dim, num_bases=2):
        super().__init__()
        self.dim = dim
        self.num_bases = num_bases

        # Identity: 始终 1 个
        self.identity = IdentityBasis(dim)

        # SO / LR / Affine: 各 K 个
        self.so_bases = nn.ModuleList([SOBasis(dim) for _ in range(num_bases)])
        self.lr_bases = nn.ModuleList([LRBasis(dim) for _ in range(num_bases)])
        self.affine_bases = nn.ModuleList([AffineBasis(dim) for _ in range(num_bases)])

        self.group_names = ['Identity', 'SO', 'LR', 'Affine']

    def add_bases(self):
        """扩展基池: 每个可扩展群各加 1 个新基原子。

        新基迁移到已有基所在设备。
        """
        target_device = next(self.so_bases[0].parameters()).device
        self.so_bases.append(SOBasis(self.dim).to(target_device))
        self.lr_bases.append(LRBasis(self.dim).to(target_device))
        self.affine_bases.append(AffineBasis(self.dim).to(target_device))
        self.num_bases += 1
        logging.info(f"GroupBasisBank: K {self.num_bases - 1} -> {self.num_bases}")

    def num_basis_per_group(self, group_name):
        if group_name == 'Identity':
            return 1
        return self.num_bases

    def forward_group(self, group_name, z, basis_weights=None):
        """按基组合权重计算群输出。

        Args:
            group_name:    'Identity' | 'SO' | 'LR' | 'Affine'
            z:             [B, N, r] 瓶颈 latent
            basis_weights: [B, K] 群内基组合权重 (Identity 忽略此参数)

        Returns:
            out: [B, N, r] 群输出 = Σ_k β_k · Basis_k(z)
        """
        if group_name == 'Identity':
            return self.identity(z)

        bases = getattr(self, {'SO': 'so_bases', 'LR': 'lr_bases',
                               'Affine': 'affine_bases'}[group_name])

        if basis_weights is None:
            # 无权重时用最后一个基 (新添加的)
            return bases[-1](z)

        # 加权组合: Σ_k β_k · Basis_k(z)
        K = len(bases)
        outputs = [bases[k](z) for k in range(K)]  # K × [B,N,r]
        stacked = torch.stack(outputs, dim=0)       # [K, B, N, r]
        w = basis_weights.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # [K, B, 1, 1]
        return (stacked * w).sum(dim=0)             # [B, N, r]

    def orthogonality_error(self):
        """所有 SO 基的正交误差之和。"""
        return sum(b.orthogonality_error() for b in self.so_bases)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CompositionRouter — 组合路由器
# ═══════════════════════════════════════════════════════════════════════════════

class CompositionRouter(nn.Module):
    """组合路由器: 输出群权重 + 群内基组合权重。

    输入: concat(cls_token, mean(tokens), std(tokens), opt_z_scores)
    输出:
      - group_weights: [B, G] 群间权重 (softmax)
      - basis_weights: [B, G, K] 群内基组合权重 (softmax per group)

    设计:
      共享 backbone MLP → 两个独立的 head:
        - group_head:    → G 个 logits
        - basis_heads:   G 个 head, 每个 → K 个 logits

    z-score 校正 (和之前一样):
      group_logit_g -= beta * stopgrad(z_g)

    Args:
        dim:          ViT 特征维度
        num_groups:   群数量 G (4)
        num_bases:    每群基数量 K
        beta:         z-score 校正强度
        tau:          softmax 温度
    """

    def __init__(self, dim, num_groups=4, num_bases=2, beta=0.1, tau=1.0):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups
        self.num_bases = num_bases
        self.beta = beta
        self.tau = tau

        # 共享 backbone
        router_input_dim = dim * 3 + num_groups  # cls + mean + std + z_scores
        router_hidden = max(dim // 2, 64)

        self.shared = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden),
            nn.GELU(),
        )

        # Group head: → G
        self.group_head = nn.Linear(router_hidden, num_groups)

        # Basis heads: Identity(g=0) 始终 1, SO/LR/Affine 各 K
        self.basis_heads = nn.ModuleList()
        for g in range(num_groups):
            if g == 0:  # Identity: 只有 1 个恒等基
                self.basis_heads.append(nn.Linear(router_hidden, 1))
            else:
                self.basis_heads.append(nn.Linear(router_hidden, num_bases))

        self._init_weights()

    def _init_weights(self):
        """稳定初始化: 最后层小权重 + 零 bias。"""
        nn.init.trunc_normal_(self.group_head.weight, std=0.02)
        nn.init.zeros_(self.group_head.bias)
        for head in self.basis_heads:
            nn.init.trunc_normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    def expand_basis_heads(self, new_num_bases):
        """扩展所有 basis head 的输出维度 (K → K+1)。

        旧列权重保留并冻结, 新列随机初始化可训练。
        """
        old_num = self.num_bases
        if new_num_bases <= old_num:
            return

        for g in range(self.num_groups):
            if g == 0:  # Identity: 始终 1, 不扩展
                continue
            old_head = self.basis_heads[g]
            new_head = nn.Linear(
                old_head.in_features, new_num_bases,
                device=old_head.weight.device,
            )
            nn.init.trunc_normal_(new_head.weight, std=0.02)
            nn.init.zeros_(new_head.bias)
            with torch.no_grad():
                new_head.weight.data[:old_num] = old_head.weight.data
                new_head.bias.data[:old_num] = old_head.bias.data

            # 旧列梯度清零, 新列可训练
            new_head.weight.requires_grad_(True)
            new_head.bias.requires_grad_(True)

            def _zero_old_grad(grad, old=old_num):
                grad[:old] = 0
                return grad
            new_head.weight.register_hook(_zero_old_grad)
            new_head.bias.register_hook(_zero_old_grad)

            self.basis_heads[g] = new_head

        self.num_bases = new_num_bases
        logging.info(f"CompositionRouter: basis K {old_num} -> {new_num_bases}")

    def forward(self, x, z_scores=None):
        """组合路由前向传播。

        Args:
            x:        [B, N, D] token 序列
            z_scores: [B, G] 逐群 z-score (可选)

        Returns:
            group_weights: [B, G] 群间软组合权重
            basis_weights: [B, G, K] 群内基软组合权重
        """
        B, N, D = x.shape
        G = self.num_groups

        # 统计特征
        cls_token = x[:, 0]
        mean_tok = x.mean(dim=1)
        std_tok = x.std(dim=1)

        if z_scores is not None:
            parts = [cls_token, mean_tok, std_tok, z_scores]
        else:
            parts = [cls_token, mean_tok, std_tok, torch.zeros(B, G, device=x.device)]

        router_input = torch.cat(parts, dim=-1)
        shared_feat = self.shared(router_input)  # [B, hidden]

        # Group weights
        group_logits = self.group_head(shared_feat)  # [B, G]
        if z_scores is not None:
            group_logits = group_logits - self.beta * z_scores.detach()
        group_weights = F.softmax(group_logits / self.tau, dim=-1)

        # Basis weights (每个群内 softmax)
        # Identity 群 (g=0): 始终 K=1, 不需要 softmax
        # SO/LR/Affine 群 (g=1,2,3): K 随扩展增长
        basis_list = []
        for g in range(G):
            if g == 0:  # Identity
                basis_list.append(torch.ones(B, 1, device=x.device))
            else:
                logits_g = self.basis_heads[g](shared_feat)  # [B, K]
                basis_list.append(F.softmax(logits_g / self.tau, dim=-1))

        # 填充到同一维度 (取最大 K)
        max_k = max(b.shape[1] for b in basis_list)
        padded = []
        for b in basis_list:
            if b.shape[1] < max_k:
                b = torch.cat([b, torch.zeros(B, max_k - b.shape[1], device=x.device)], dim=-1)
            padded.append(b)
        basis_weights = torch.stack(padded, dim=1)  # [B, G, max_K]

        return group_weights, basis_weights


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GroupAwareAE — 逐群自编码器
# ═══════════════════════════════════════════════════════════════════════════════

class GroupAwareAE(nn.Module):
    """逐群自编码器: G 个 encoder→decoder 对。

    L_group_RD = Σ_g p(g|h) · MSE(AE_g(z), z)
    """

    def __init__(self, dim, rd_dim=None, num_groups=4):
        super().__init__()
        self.dim = dim
        self.rd_dim = rd_dim or max(dim // 4, 4)
        self.num_groups = num_groups

        self.encoders = nn.ModuleList([
            nn.Linear(dim, self.rd_dim) for _ in range(num_groups)
        ])
        self.decoders = nn.ModuleList([
            nn.Linear(self.rd_dim, dim) for _ in range(num_groups)
        ])
        for enc, dec in zip(self.encoders, self.decoders):
            nn.init.kaiming_uniform_(enc.weight, a=math.sqrt(5))
            nn.init.zeros_(enc.bias)
            nn.init.kaiming_uniform_(dec.weight, a=math.sqrt(5))
            nn.init.zeros_(dec.bias)

    def forward_all(self, z):
        """z: [B, dim] → [G, B, dim]"""
        out = []
        for g in range(self.num_groups):
            out.append(self.decoders[g](self.encoders[g](z)))
        return torch.stack(out, dim=0)

    def compute_group_rd_loss(self, z, group_weights):
        B, D = z.shape
        G = self.num_groups
        all_rec = self.forward_all(z)  # [G, B, D]
        per_group_loss = torch.zeros(B, G, device=z.device)
        for g in range(G):
            per_group_loss[:, g] = F.mse_loss(all_rec[g], z, reduction='none').mean(dim=-1)
        rd_loss = (per_group_loss * group_weights).sum(dim=-1)
        return rd_loss, per_group_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RunningRecords — 逐群统计
# ═══════════════════════════════════════════════════════════════════════════════

class RunningRecords:
    """运行统计缓冲区, 维护 RD 损失的均值和标准差。"""

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
# 6. GroupBasisAdapter — 完整群基组合适配器
# ═══════════════════════════════════════════════════════════════════════════════

class GroupBasisAdapter(nn.Module):
    """群基组合适配器。

    数据流:
      1. z = down_proj(LN(x))                          — 瓶颈投影
      2. group_w, basis_w = CompositionRouter(x, z_scores) — 组合路由
      3. For each group g:
           out_g = Σ_k basis_w[g,k] · Basis_{g,k}(z)   — 群内基组合
      4. z_G = Σ_g group_w[g] · out_g                  — 群间组合
      5. a = up_proj(z_G)                               — 输出投影
      6. RD: L = Σ_g group_w[g] · MSE(AE_g(z), z)      — 重建检测

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
        num_groups = getattr(config, 'num_groups', 4)
        num_bases = getattr(config, 'init_bases', 2)

        # 瓶颈投影
        self.down_proj = nn.Linear(d_model, bottleneck)
        self.up_proj = nn.Linear(bottleneck, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gamma = nn.Parameter(torch.zeros(1))

        # 群基原子池
        self.basis_bank = GroupBasisBank(bottleneck, num_bases=num_bases)

        # 组合路由器
        self.router = CompositionRouter(
            d_model, num_groups=num_groups, num_bases=num_bases,
            beta=getattr(config, 'router_beta', 0.1),
            tau=getattr(config, 'router_tau', 1.0),
        )

        # 逐群 AE/RD (仅深层)
        self.not_addition_layer = (
            layer_id < config.adapt_start_layer
            or layer_id > config.adapt_end_layer
        )
        if not self.not_addition_layer:
            self.group_ae = GroupAwareAE(
                bottleneck,
                rd_dim=getattr(config, 'rd_dim', 32),
                num_groups=num_groups,
            )
            self.per_group_records = [
                RunningRecords(max_len=getattr(config, 'buffer_size', 500))
                for _ in range(num_groups)
            ]
        else:
            self.group_ae = None
            self.per_group_records = None

        self.newly_added = True
        self._init_weights()

        # ── 截面保持 (Section Preservation) ──
        # 存储上一任务的群路由快照, 用于保护旧截影
        self._protect_old = False
        self._protect_threshold = getattr(config, 'protect_threshold', 2.0)
        self.stored_group_weights = None   # [G] 旧群权重分布
        self.stored_basis_weights = None   # [G, K] 旧基权重分布
        self._capture_count = 0            # 快照捕获计数器
        self._capture_gw_sum = None        # 累积 group_weights
        self._capture_bw_sum = None        # 累积 basis_weights

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)   # 零初始化
        nn.init.zeros_(self.up_proj.bias)

    def _compute_z_scores(self, per_group_loss, detach=True):
        B, G = per_group_loss.shape
        z_scores = torch.zeros(B, G, device=per_group_loss.device)
        for g in range(G):
            rec = self.per_group_records[g]
            if rec.length > 2:
                loss_g = per_group_loss[:, g].detach() if detach else per_group_loss[:, g]
                z_scores[:, g] = torch.abs((loss_g - rec.mean) / rec.stddev)
        return z_scores

    def forward(self, x, compute_rd=True):
        """前向传播。

        Args:
            x:          [B, N, d_model]
            compute_rd: 是否计算 RD

        Returns:
            dict: func_out, rd_loss, z_scores, group_weights, basis_weights, added
        """
        B, N, D = x.shape
        x_norm = self.norm(x)

        # Step 1: 瓶颈投影
        z = self.down_proj(x_norm)  # [B, N, r]

        # Step 2: 路由准备
        use_ae = (compute_rd and not self.not_addition_layer
                  and self.group_ae is not None)
        if use_ae:
            z_pooled = z.mean(dim=1)
            uniform_w = torch.ones(B, len(self.basis_bank.group_names),
                                   device=z.device) / len(self.basis_bank.group_names)
            _, per_group_rd = self.group_ae.compute_group_rd_loss(z_pooled, uniform_w)
            z_scores = self._compute_z_scores(per_group_rd)
            group_weights, basis_weights = self.router(x_norm, z_scores=z_scores)
        else:
            group_weights, basis_weights = self.router(x_norm, z_scores=None)

        # ── 截面保持: z-score 低的群强制走旧路由, 保护旧截影 ──
        if self._protect_old and self.stored_group_weights is not None:
            # 强制计算 z-score (即使 compute_rd=False, 也需要用于门控)
            if not use_ae:
                z_pooled = z.mean(dim=1)
                uniform_w = torch.ones(B, len(self.basis_bank.group_names),
                                       device=z.device) / len(self.basis_bank.group_names)
                _, per_group_rd = self.group_ae.compute_group_rd_loss(z_pooled, uniform_w)
                z_scores_gate = self._compute_z_scores(per_group_rd, detach=True)
            else:
                z_scores_gate = z_scores

            # 逐样本门控: z < 阈值 → 该群可被旧截影解释 → 锁定路由
            protect_mask = (z_scores_gate < self._protect_threshold).float()  # [B, G]
            # 存储的旧权重作为锚点 (stopgrad 防止被更新)
            stored_gw = self.stored_group_weights.to(x.device).unsqueeze(0)  # [1, G]
            group_weights = group_weights * (1 - protect_mask) + stored_gw.detach() * protect_mask
            # basis_weights 同理: 受保护的群用旧基组合
            if self.stored_basis_weights is not None:
                stored_bw = self.stored_basis_weights.to(x.device).unsqueeze(0)  # [1, G, K]
                # pad stored_bw if K changed
                cur_K = basis_weights.shape[2]
                if stored_bw.shape[2] < cur_K:
                    pad = torch.zeros(1, stored_bw.shape[1], cur_K - stored_bw.shape[2],
                                     device=x.device)
                    stored_bw = torch.cat([stored_bw, pad], dim=-1)
                pm = protect_mask.unsqueeze(-1)  # [B, G, 1]
                basis_weights = basis_weights * (1 - pm) + stored_bw.detach() * pm

        # Step 3-4: 群内基组合 + 群间组合
        group_outputs = []
        for g, gn in enumerate(self.basis_bank.group_names):
            # 取该群的 basis_weights: [B, K]
            bw_g = basis_weights[:, g, :]  # [B, K]
            out_g = self.basis_bank.forward_group(gn, z, basis_weights=bw_g)
            group_outputs.append(out_g)

        stacked = torch.stack(group_outputs, dim=0)  # [G, B, N, r]
        w = group_weights.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # [G, B, 1, 1]
        z_G = (stacked * w).sum(dim=0)  # [B, N, r]

        # Step 5: 输出投影
        a = self.up_proj(z_G)  # [B, N, D]

        # Step 6: RD 损失
        rd_loss = torch.tensor(0.0, device=x.device)
        z_scores_out = torch.zeros(B, len(self.basis_bank.group_names), device=x.device)

        if use_ae:
            z_pooled = z.mean(dim=1)
            rd_loss, per_group_rd = self.group_ae.compute_group_rd_loss(
                z_pooled, group_weights
            )
            rd_loss = rd_loss.mean()
            if self.training:
                for g in range(len(self.basis_bank.group_names)):
                    self.per_group_records[g].add_record(per_group_rd[:, g])
            z_scores_out = self._compute_z_scores(per_group_rd, detach=False)

        # ── 快照捕获: 累积路由权重用于截面保护 ──
        if hasattr(self, '_capture_count') and self._capture_count >= 0:
            gw_mean = group_weights.mean(dim=0).detach()  # [G]
            bw_mean = basis_weights.mean(dim=0).detach()  # [G, K]
            self._capture_gw_sum = (self._capture_gw_sum + gw_mean
                                    if self._capture_count > 0 else gw_mean)
            self._capture_bw_sum = (self._capture_bw_sum + bw_mean
                                    if self._capture_count > 0 else bw_mean)
            self._capture_count += 1

        return {
            "func_out": a,
            "rd_loss": rd_loss,
            "z_scores": z_scores_out,
            "group_weights": group_weights,
            "basis_weights": basis_weights,
            "added": False,
        }

    def add_bases(self):
        """扩展基池: 每个可扩展群各加 1 个新基。"""
        self.basis_bank.add_bases()
        self.router.expand_basis_heads(self.basis_bank.num_bases)

    def orthogonality_error(self):
        return self.basis_bank.orthogonality_error()

    def start_snapshot_capture(self):
        """开始捕获路由快照 (重置累积器)。"""
        self._capture_count = 0
        self._capture_gw_sum = None
        self._capture_bw_sum = None

    def finish_snapshot_capture(self):
        """结束捕获, 将累积均值存储为保护快照。"""
        if self._capture_count > 0 and self._capture_gw_sum is not None:
            self.stored_group_weights = (self._capture_gw_sum / self._capture_count).detach().cpu()
            self.stored_basis_weights = (self._capture_bw_sum / self._capture_count).detach().cpu()
            logging.info(f"Layer {self.layer_id}: stored routing snapshot "
                         f"(gw={self.stored_group_weights.shape}, bw={self.stored_basis_weights.shape})")
        self._capture_count = 0
        self._capture_gw_sum = None
        self._capture_bw_sum = None

    def set_protect_old(self, enable):
        self._protect_old = enable

    # ── 在 __init__ 调用 _init_snapshot ──


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GroupBasisModules — 层级管理器
# ═══════════════════════════════════════════════════════════════════════════════

class GroupBasisModules(nn.Module):
    """层级群基组合管理器。

    管理:
      - 基池 (GroupBasisBank)
      - 扩展检测 (z-score + 持续性)
      - 任务管理 (freeze, end_of_task)
    """

    def __init__(self, config, layer_id, writer=None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.writer = writer
        self.adapt_start_layer = config.adapt_start_layer
        self.adapt_end_layer = config.adapt_end_layer

        # 扩展检测状态
        self.detecting_outlier = False
        self.added_for_task = False
        self.newly_added = True

        # 多 batch 持续性检测
        expansion_patience = getattr(config, 'expansion_patience', 3)
        self._z_score_accum = torch.zeros(4)
        self._z_score_count = 0
        self._expansion_patience = expansion_patience
        self._expansion_candidate = -1

        # 初始化适配器
        self.adapters: List[GroupBasisAdapter] = nn.ModuleList()
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
        new_adapter = GroupBasisAdapter(self.config, self.layer_id).to(self._device())
        self.newly_added = True
        self.added_for_task = True
        self.adapters.append(new_adapter)
        if not initialize:
            logging.info(f"GroupBasisAdapter {self.layer_id} 已添加")

    def add_bases(self):
        """扩展基池: 每群各加 1 个新基。"""
        if self.adapters:
            adapter = self.adapters[-1]
            adapter.add_bases()
            bank = adapter.basis_bank
            for basis in [bank.so_bases[-1], bank.lr_bases[-1], bank.affine_bases[-1]]:
                for param in basis.parameters():
                    param.requires_grad = True
            self.expansion_count += 1

    def start_snapshot_capture(self):
        for adapter in self.adapters:
            adapter.start_snapshot_capture()

    def finish_snapshot_capture(self):
        for adapter in self.adapters:
            adapter.finish_snapshot_capture()

    def protect_old_sections(self):
        """启用截面保护: 训练时锁定旧截影的群路由。"""
        for adapter in self.adapters:
            adapter.set_protect_old(True)

    def unprotect_sections(self):
        """关闭截面保护 (评估/检测时不需要)。"""
        for adapter in self.adapters:
            adapter.set_protect_old(False)

    def forward(self, x, group_info=None):
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
                "group_weights": adapter_out["group_weights"],
                "basis_weights": adapter_out["basis_weights"],
                "added": False,
            }

        compute_rd = self.detecting_outlier or getattr(self, '_training_rd', False)
        adapter = self.adapters[-1]
        adapter_out = adapter(x, compute_rd=compute_rd)

        # 扩展检测
        added = False
        if self.detecting_outlier and not self.added_for_task:
            z_scores = adapter_out.get("z_scores")
            if z_scores is not None and z_scores.shape[1] > 0:
                batch_z_mean = z_scores.mean(dim=0).detach().cpu()
                self._z_score_accum = (
                    self._z_score_accum * self._z_score_count + batch_z_mean
                ) / (self._z_score_count + 1)
                self._z_score_count += 1

                best_idx = self._z_score_accum.argmax().item()
                max_z = self._z_score_accum[best_idx].item()

                if max_z > self.config.exp_threshold:
                    if (self._expansion_candidate == best_idx
                            and self._z_score_count >= self._expansion_patience):
                        self.add_bases()
                        self.added_for_task = True
                        added = True
                        logging.info(
                            f"Block {self.layer_id}: 添加新基 (K={adapter.basis_bank.num_bases}) "
                            f"(z={max_z:.3f})"
                        )
                        self._z_score_accum.zero_()
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
            "group_weights": adapter_out["group_weights"],
            "basis_weights": adapter_out["basis_weights"],
            "added": added,
        }

    # ── 任务管理 ──

    def end_of_task_training(self):
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
        for adapter in self.adapters:
            for param in adapter.down_proj.parameters():
                param.requires_grad = False
            for param in adapter.up_proj.parameters():
                param.requires_grad = False
            adapter.gamma.requires_grad_(False)
            for param in adapter.basis_bank.parameters():
                param.requires_grad = False
            for param in adapter.router.parameters():
                param.requires_grad = False

    def freeze_rd(self):
        for adapter in self.adapters:
            if adapter.group_ae is not None:
                for param in adapter.group_ae.parameters():
                    param.requires_grad = False
                if adapter.per_group_records:
                    for rec in adapter.per_group_records:
                        rec.updating = False
