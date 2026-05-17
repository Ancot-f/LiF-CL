"""
Geometry-Aware SEMA with Group-MoE and Shared Mamba Flow
=========================================================

核心设计理念 (suggest.md):
  将持续学习建模为"几何感知的语义结构保持"问题。
  - 固定的预训练 ViT 提供稳定的语义基空间
  - 固定的 GroupBank 提供候选几何先验 (Identity, SO, LR, Affine)
  - Group-MoE 选择和扩展局部群特定专家来吸收新任务变化
  - 共享的 MambaFlow 建模几何条件化后的语义输运 (不扩展)
  - Group-aware AE/RD 判断当前样本是否能被现有群条件知识解释

数据流 (section 5-6):
  z = W_down LN(h)              -- 瓶颈投影
  pi = GroupRouter(z, RD_stats) -- 几何路由 (群概率 + 专家概率)
  z^G = sum_g pi_g * T_g(z)    -- Group-MoE 混合
  m = SharedMambaFlow(z^G)      -- 几何条件化语义流 (不扩展)
  a = W_up m                    -- 输出投影
  h_out = h + gamma * a         -- 残差连接

三层损失 (section 11):
  L_total = L_cls + lambda_1 * L_geo_rd + lambda_2 * L_sem

组件清单:
  - selective_scan:        选择性 SSM 扫描 (S6 核心算子)
  - SharedMambaFlow:       共享 Mamba 语义流算子 (不扩展)
  - IdentityExpert:        恒等专家 T_ID(z) = 0
  - SOExpert:              SO(r) 旋转专家 z_SO = z @ R
  - LRExpert:              低秩专家 z_LR = z + z @ A @ B
  - AffineExpert:          仿射专家 z_Affine = z @ W + b
  - GroupBank:             固定群类型 + 可扩展群特定专家
  - HierarchicalRouter:    层次几何路由 (GroupRouter -> ExpertRouter)
  - GroupAwareAE:          群感知自编码器 (逐群 RD)
  - RunningRecords:        逐群运行统计缓冲区
  - GroupMoEAdapter:       完整的 Group-MoE 适配器
  - GeometrySEMAModules:   层级管理器 (替换 SEMAModules)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import math
import logging


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Selective Scan — Mamba S6 核心算子
# ═══════════════════════════════════════════════════════════════════════════════

def selective_scan(u, delta, A, B, C, D):
    """选择性 SSM 扫描 (sequential 实现, O(L) 复杂度)。

    实现 S6 状态空间模型的核心递推:
      h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t
      y_t = C_t * h_t + D * u_t

    其中:
      - A: 对角状态矩阵 (HiPPO-LegS 初始化, 固定结构)
      - delta: 输入依赖的步长 (选择性机制的关键)
      - B, C: 输入依赖的投影矩阵
      - D: 跳跃连接参数

    Args:
        u:     [B, L, D]  输入序列
        delta: [B, L, D]  每通道步长 (通过 Softplus 保证正)
        A:     [D, N]     对角状态矩阵 (N = d_state)
        B:     [B, L, N]  输入投影
        C:     [B, L, N]  输出投影
        D:     [D]        跳跃连接

    Returns:
        y: [B, L, D] 输出序列

    Note:
        此实现为顺序扫描, O(L) 时间复杂度。
        生产环境中可用关联扫描 (associative scan) 实现 O(log L) 并行化。
    """
    Bsz, L, D = u.shape
    N = A.shape[1]  # 状态维度 d_state

    # deltaA: [B, L, D, N] — 离散化状态转移矩阵 exp(delta * A)
    deltaA = torch.exp(delta.unsqueeze(-1) * A)

    # deltaB_u: [B, L, D, N] — 输入项 delta * B * u
    # B: [B, L, N] -> unsqueeze(-2) -> [B, L, 1, N] 保证与 D 维度正确广播
    deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(-2) * u.unsqueeze(-1)

    # ── 顺序递推扫描 ──
    h = torch.zeros(Bsz, D, N, device=u.device, dtype=u.dtype)
    ys = []
    for i in range(L):
        h = deltaA[:, i] * h + deltaB_u[:, i]
        y_i = (h * C[:, i].unsqueeze(-2)).sum(dim=-1)
        ys.append(y_i)

    y = torch.stack(ys, dim=1) + u * D
    return y


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SharedMambaFlow — 共享几何条件化语义流算子
# ═══════════════════════════════════════════════════════════════════════════════

class SharedMambaFlow(nn.Module):
    """共享的 Mamba 语义流算子 —— 不可扩展的稳定传输算子。

    设计理念 (suggest.md section 6):
      - Mamba 在几何条件化后对 token 序列建模语义状态演化
      - 不替代 ViT 主干, 不存储任务特定知识, 不自由重写语义空间
      - 作为稳定的"流/联络"算子, 在所有任务间共享

    架构 (简化 S6 风格):
      1. 输入投影 + 门控分离:  x -> in_proj -> (x_ssm, gate)
      2. 1D 深度可分离卷积:   捕捉局部 token 上下文
      3. SSM 参数投影:        x -> (dt, B, C)
      4. 选择性扫描:          selective_scan(x, delta, A, B, C, D)
      5. SiLU 门控:          y = y * silu(gate)
      6. 输出投影 + 残差:     out = out_proj(y) + gamma * x

    Args:
        dim:      输入/输出维度 (瓶颈维度 r, 如 16 或 32)
        d_state:  SSM 状态维度 N (默认 16)
        d_conv:   1D 卷积核大小 (默认 4)
        expand:   内部扩展因子 (默认 2, inner_dim = dim * expand)
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        inner_dim = dim * expand

        # ── 1. 输入投影 + 门控 ──
        # 将输入投影到 inner_dim 并分为两支: x_ssm (进入 SSM) 和 gate (门控信号)
        self.in_proj = nn.Linear(dim, inner_dim * 2)

        # ── 2. 1D 深度可分离卷积 ──
        # groups=inner_dim 保证每个通道独立卷积, 捕捉局部 token 间关系
        # padding = d_conv - 1 保证因果性 (只看过去)
        self.conv1d = nn.Conv1d(
            inner_dim, inner_dim, d_conv,
            groups=inner_dim,          # 深度可分离
            padding=d_conv - 1,        # 因果卷积
        )

        # ── 3. SSM 参数投影 ──
        # 从 x 投影得到: dt_rank (低秩步长参数) + B (d_state) + C (d_state)
        dt_rank = max(1, math.ceil(dim / 16))
        self.x_proj = nn.Linear(inner_dim, dt_rank + d_state * 2, bias=False)

        # ── 4. Delta 投影 ──
        # dt_rank -> inner_dim, 经过 Softplus 保证 delta > 0
        self.dt_proj = nn.Sequential(
            nn.Linear(dt_rank, inner_dim),
            nn.Softplus(),
        )

        # ── 5. 对角状态矩阵 A ──
        # HiPPO-LegS 初始化: A[n] = -sqrt(n+1), 捕获长程依赖
        # 存储为 log(-A) 保证 A 始终为负 (稳定离散化)
        A = torch.empty(inner_dim, d_state)
        for i in range(d_state):
            A[:, i] = -((i + 1) ** 0.5) * torch.ones(inner_dim)
        self.A_log = nn.Parameter(torch.log(-A))  # 可学习

        # ── 6. 跳跃连接 D ──
        self.D = nn.Parameter(torch.ones(inner_dim))

        # ── 7. 输出投影 ──
        # inner_dim -> dim, 恢复到瓶颈维度
        self.out_proj = nn.Linear(inner_dim, dim)

        # ── 8. 残差门控 ──
        # 零初始化保证训练初期 Mamba 不干扰预训练特征
        self.gamma = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        """权重初始化: Kaiming Uniform + Zero Bias 保证训练稳定性。"""
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.in_proj.bias)
        nn.init.kaiming_uniform_(self.x_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        """前向传播: 几何条件化 latent -> 语义流 -> 输出。

        Args:
            x: [B, L, dim]  群混合后的几何条件化 latent tokens

        Returns:
            out: [B, L, dim]  Mamba 语义流输出
        """
        B, L, D = x.shape

        # 1. 输入投影 + 门控分离
        x_and_res = self.in_proj(x)                 # [B, L, inner_dim * 2]
        x_ssm, gate = x_and_res.chunk(2, dim=-1)    # 各 [B, L, inner_dim]

        # 2. 1D 因果卷积: 注入局部 token 上下文
        x_conv = self.conv1d(x_ssm.transpose(1, 2))       # [B, inner_dim, L + pad]
        x_conv = x_conv[:, :, :L].transpose(1, 2)          # [B, L, inner_dim]
        x_conv = F.silu(x_conv)                             # 非线性激活

        # 3. SSM 参数投影: x -> (dt, B, C)
        ssm_params = self.x_proj(x_conv)                    # [B, L, dt_rank + 2*d_state]
        dt_rank = self.x_proj.out_features - self.d_state * 2
        dt, B_ssm, C_ssm = ssm_params.split(
            [dt_rank, self.d_state, self.d_state], dim=-1
        )

        # 4. Delta 投影: dt_rank -> inner_dim, Softplus 保证正
        delta = self.dt_proj(dt)                            # [B, L, inner_dim]

        # 5. 对角状态矩阵 A: 从 log 空间恢复
        A = -torch.exp(self.A_log)                          # [inner_dim, d_state]

        # 6. 选择性扫描: S6 核心递推
        y = selective_scan(
            x_conv, delta, A, B_ssm, C_ssm, self.D
        )  # [B, L, inner_dim]

        # 7. SiLU 门控: 选择性信息过滤
        y = y * F.silu(gate)

        # 8. 输出投影 + 残差连接
        out = self.out_proj(y)                              # [B, L, dim]
        out = out + self.gamma * x                          # 残差 (gamma 初始为 0)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Group Experts — 群特定专家模块
# ═══════════════════════════════════════════════════════════════════════════════

class IdentityExpert(nn.Module):
    """恒等专家: T_ID(z) = 0 (零残差, 主路径已经携带 u_l)。

    设计理由 (suggest.md section 5.1):
      不需要额外的恒等映射, 因为残差连接 x = u_l + gamma * a 已经保证
      信号可以直接通过。输出零意味着 "这个群不贡献任何变换"。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, z):
        """返回零张量, 保持输入不变 (通过残差连接)。"""
        return torch.zeros_like(z)


class SOExpert(nn.Module):
    """SO(r) 旋转专家: z_SO = z @ R, R 被鼓励趋向 SO(r)。

    几何意义:
      SO(r) 群描述 r 维空间中的旋转。在瓶颈空间中,
      旋转变换保持 token 间的角度关系 (内积不变),
      适合建模需要保持语义相似性的任务变化。

    约束方式:
      不强制 R ∈ SO(r) (会破坏梯度流), 而是通过损失函数软约束:
        L_geo = ||R^T R - I||^2_F  (section 11.2)

    初始化:
      R = I + 小噪声, 保证初始接近恒等变换。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 初始化为单位矩阵 + 小随机扰动
        self.R = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01)

    def forward(self, z):
        """应用旋转变换: z -> z @ R。

        Args:
            z: [B, N, dim] 瓶颈空间 token

        Returns:
            z_SO: [B, N, dim] 旋转后 token
        """
        return z @ self.R

    def orthogonality_error(self):
        """计算 ||R^T R - I||^2_F, 用于 SO 正交约束损失。

        Returns:
            scalar: Frobenius 范数的平方
        """
        RTR = self.R.T @ self.R
        eye = torch.eye(self.dim, device=self.R.device)
        return torch.norm(RTR - eye, p='fro') ** 2


class LRExpert(nn.Module):
    """低秩专家: z_LR = z + z @ A @ B, A ∈ R^{r×k}, B ∈ R^{k×r}, k << r。

    几何意义:
      低秩修正可视为在 r 维空间中的一个 k 维子空间中进行变换。
      这类似于在纤维上的局部截面中进行微小调整,
      只改变少数方向上的特征, 保持大部分语义结构不变。

    参数效率:
      参数量 = 2 * r * k, 如 r=16, k=4: 2*16*4 = 128 参数
    """

    def __init__(self, dim, rank=None):
        super().__init__()
        self.dim = dim
        self.rank = rank or max(1, dim // 4)  # k = dim/4, 如 16->4
        # 小初始化保证初期接近恒等变换
        self.A = nn.Parameter(torch.randn(dim, self.rank) * 0.01 / math.sqrt(self.rank))
        self.B = nn.Parameter(torch.randn(self.rank, dim) * 0.01 / math.sqrt(self.rank))

    def forward(self, z):
        """低秩修正: z -> z + z @ A @ B。

        Args:
            z: [B, N, dim]

        Returns:
            z_LR: [B, N, dim]
        """
        return z + z @ self.A @ self.B


class AffineExpert(nn.Module):
    """仿射专家: z_Affine = z @ W + b。

    几何意义:
      仿射变换是最大灵活性的群操作 (包括旋转、缩放、平移)。
      提供最强的适应能力, 但也最容易过拟合。
      建议谨慎使用, 优先使用 SO 和 LR 专家。

    初始化:
      W = I + 小噪声, b = 0, 保证初始接近恒等变换。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.W = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        """仿射变换: z -> z @ W + b。

        Args:
            z: [B, N, dim]

        Returns:
            z_Affine: [B, N, dim]
        """
        return z @ self.W + self.b


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GroupBank — 固定群类型 + 可扩展群特定专家
# ═══════════════════════════════════════════════════════════════════════════════

class GroupBank(nn.Module):
    """固定群候选库, 包含可扩展的群特定专家。

    群类型 (suggest.md section 4):
      - Identity:   单一零残差路径 (不可扩展, 不需要扩展)
      - SO:         可扩展 SO(r) 旋转专家: SO_0, SO_1, SO_2, ...
      - LR:         可扩展低秩专家: LR_0, LR_1, ...
      - Affine:     可扩展仿射专家: Affine_0, Affine_1, ...

    MambaFlow 不在此 GroupBank 中 —— 它是独立共享的语义流算子,
    在 Group-MoE 混合后应用 (suggest.md section 6)。

    扩展策略 (section 9):
      - SO, LR, Affine: 可进行群特定专家扩展
      - Identity: 不可扩展 (始终只有一个零残差路径)
      - 群类型本身: v1 版本不扩展

    Args:
        dim:               瓶颈维度 r (如 16)
        expandable_groups: 可扩展的群类型列表
    """

    def __init__(self, dim, expandable_groups=('SO', 'LR', 'Affine')):
        super().__init__()
        self.dim = dim
        self.expandable_groups = expandable_groups

        # ── 专家工厂: 群名称 -> 构造函数 ──
        self._expert_factory = {
            'Identity': lambda: IdentityExpert(dim),
            'SO':       lambda: SOExpert(dim),
            'LR':       lambda: LRExpert(dim),
            'Affine':   lambda: AffineExpert(dim),
        }

        # ── 群 -> ModuleList[expert] ──
        # 每个群初始包含 1 个专家
        self.groups: Dict[str, nn.ModuleList] = nn.ModuleDict()
        for group_name in ['Identity', 'SO', 'LR', 'Affine']:
            self.groups[group_name] = nn.ModuleList()
            self._add_expert_to_group(group_name)

    def _add_expert_to_group(self, group_name):
        """在指定群中添加一个新专家，自动迁移到已有专家的设备。"""
        expert = self._expert_factory[group_name]()
        # 将新专家迁移到已有专家的设备 (处理扩展时 CPU -> GPU 迁移)
        existing = self.groups[group_name]
        if len(existing) > 0:
            target_device = next(existing[0].parameters()).device
            expert = expert.to(target_device)
        existing.append(expert)
        return expert

    def add_expert(self, group_name):
        """向可扩展群添加新专家 (用于持续学习扩展)。

        Args:
            group_name: 群名称 ('SO', 'LR', 'Affine')

        Returns:
            bool: 是否添加成功
        """
        if group_name not in self.expandable_groups:
            logging.warning(
                f"群 '{group_name}' 不可扩展。跳过扩展。"
            )
            return False
        self._add_expert_to_group(group_name)
        logging.info(
            f"群 '{group_name}' 添加新专家 "
            f"(当前 {len(self.groups[group_name])} 个专家)"
        )
        return True

    def get_expert(self, group_name, expert_idx):
        """获取指定群中的指定专家。"""
        return self.groups[group_name][expert_idx]

    def num_experts(self, group_name):
        """获取指定群中的专家数量。"""
        return len(self.groups[group_name])

    def forward_group(self, group_name, z, expert_weights=None):
        """对指定群应用专家变换 (支持加权混合)。

        Args:
            group_name:     群名称
            z:              [B, N, dim] 瓶颈 latent tokens
            expert_weights: [B, num_experts] 专家权重, None 表示只用最后一个

        Returns:
            out:            [B, N, dim] 群输出
            expert_outputs: 各专家输出的列表 (或 None)
        """
        experts = self.groups[group_name]

        if expert_weights is None or len(experts) == 1:
            # 单专家: 直接使用最后一个 (或唯一一个) 专家
            return experts[-1](z), None

        # 多专家加权混合
        expert_outs = []
        for expert in experts:
            expert_outs.append(expert(z))

        # stacked: [E, B, N, dim]
        stacked = torch.stack(expert_outs, dim=0)
        # expert_weights: [B, E] -> [E, B, 1, 1]
        w = expert_weights.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
        combined = (stacked * w).sum(dim=0)  # [B, N, dim]
        return combined, expert_outs

    def orthogonality_error(self):
        """所有 SO 专家的正交误差之和, 用于 L_geo 损失。"""
        err = 0.0
        for expert in self.groups['SO']:
            err = err + expert.orthogonality_error()
        return err


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HierarchicalRouter — 层次几何路由
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchicalRouter(nn.Module):
    """层次几何路由器: GroupRouter -> ExpertRouter。

    设计 (suggest.md section 7):

    路由输入 (统计特征拼接):
      router_input = concat(
        cls_token,              # [D] 分类 token
        mean(tokens),           # [D] 全局平均
        std(tokens),            # [D] 全局标准差
        group_wise_RD_z_scores, # [G] 逐群 RD z-score
        optional_group_usage    # [G] 群使用统计
      )

    群得分 (z-score 校正):
      score_g = MLP(h)_g - beta * stopgrad(z_g)
      p(g|h) = softmax(score_g / tau)

    最终专家权重 (层次路由):
      w_{g,e} = p(g|h) * p(e|g,h)

    稀疏路由 (训练/推理):
      - 训练: soft routing 或 top-2 routing
      - 推理: top-1 或 top-2 routing

    Args:
        dim:            ViT 特征维度 D (768)
        num_groups:     群数量 G (4: Identity, SO, LR, Affine)
        beta:           z-score 校正强度
        tau:            路由温度 (越小越尖锐)
        router_hidden:  路由 MLP 隐藏维度
    """

    def __init__(self, dim, num_groups=4, beta=0.1,
                 tau=1.0, router_hidden=None):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups
        self.beta = beta   # z-score 校正系数
        self.tau = tau     # softmax 温度

        # ── 路由输入维度计算 ──
        # cls_token(D) + mean(D) + std(D) + z_scores(G) + usage(G)
        router_input_dim = dim * 3 + num_groups * 2
        router_hidden = router_hidden or max(dim // 2, 64)

        # ── GroupRouter: 输入 -> G 个群 logits ──
        self.group_router = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, num_groups),
        )

        # ── ExpertRouter: 每个可扩展群有独立的路由器 ──
        # 键为群名称, 值为该群的专家路由器
        self.expert_routers = nn.ModuleDict()

        self._init_weights()

    def _init_weights(self):
        """稳定初始化: 最后层小权重 + 零 bias, 使初始路由接近均匀。"""
        last = self.group_router[-1]
        nn.init.trunc_normal_(last.weight, std=0.02)
        nn.init.zeros_(last.bias)

    def ensure_expert_router(self, group_name, num_experts):
        """创建或扩展指定群的专家路由器。

        扩展时保留旧路由权重 (冻结旧列, 新列可训练):
          - 将输出层从 [hidden, old_N] 扩展到 [hidden, new_N]
          - 旧权重复制, 新权重随机初始化
          - 旧列梯度冻结, 只有新列参与训练

        Args:
            group_name:  群名称 ('SO', 'LR', 'Affine')
            num_experts: 该群当前专家数量
        """
        router_input_dim = self.group_router[0].in_features
        router_hidden = self.group_router[0].out_features

        if group_name not in self.expert_routers:
            # 首次创建
            router = nn.Sequential(
                nn.Linear(router_input_dim, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, num_experts),
            )
            self.expert_routers[group_name] = router
            nn.init.trunc_normal_(router[-1].weight, std=0.02)
            nn.init.zeros_(router[-1].bias)
        else:
            # 扩展已有路由器: 增加输出维度
            old_router = self.expert_routers[group_name]
            old_output = old_router[-1]  # nn.Linear(hidden, old_num)
            old_num = old_output.out_features
            if num_experts > old_num:
                # 创建扩展的输出层
                new_output = nn.Linear(
                    old_output.in_features, num_experts,
                    device=old_output.weight.device,
                )
                # 复制旧权重 + 随机初始化新列
                nn.init.trunc_normal_(new_output.weight, std=0.02)
                nn.init.zeros_(new_output.bias)
                with torch.no_grad():
                    new_output.weight.data[:old_num] = old_output.weight.data
                    new_output.bias.data[:old_num] = old_output.bias.data
                # 旧列梯度清零 hook: 只允许新列学习, 旧列保持不变
                # 先 requires_grad=True (hook 需要), 再注册清零 hook
                new_output.weight.requires_grad_(True)
                new_output.bias.requires_grad_(True)

                def _zero_old_grad(grad):
                    grad[:old_num] = 0
                    return grad
                new_output.weight.register_hook(_zero_old_grad)
                new_output.bias.register_hook(_zero_old_grad)

                # 替换
                old_router[-1] = new_output
                logging.info(
                    f"ExpertRouter '{group_name}': {old_num} -> {num_experts} experts"
                )

    def _build_router_input(self, x, z_scores=None, group_usage=None):
        """构建路由器输入特征。

        从 token 序列中提取统计特征, 拼接 RD z-score 和群使用统计,
        为路由器提供丰富的上下文信息。

        Args:
            x:           [B, N, D] token 序列
            z_scores:    [B, G] 逐群 RD z-score (可为 None)
            group_usage: [B, G] 群使用统计 (可为 None)

        Returns:
            router_input: [B, router_input_dim]
        """
        B, N, D = x.shape

        # 从 token 序列提取统计特征
        cls_token = x[:, 0]        # [B, D] 分类 token
        mean_tok = x.mean(dim=1)   # [B, D] 空间均值
        std_tok = x.std(dim=1)     # [B, D] 空间标准差

        parts = [cls_token, mean_tok, std_tok]

        # RD z-score: 反映当前样本在各群下的异常程度
        if z_scores is not None:
            parts.append(z_scores)
        else:
            parts.append(torch.zeros(B, self.num_groups, device=x.device))

        # 群使用统计: 反映各群的历史使用情况
        if group_usage is not None:
            parts.append(group_usage)
        else:
            parts.append(torch.zeros(B, self.num_groups, device=x.device))

        return torch.cat(parts, dim=-1)

    def forward(self, x, z_scores=None, group_usage=None,
                group_expert_counts=None):
        """层次路由前向传播。

        1. GroupRouter: 选择群类型
        2. ExpertRouter: 在选中群中选择专家
        3. 最终权重: w_{g,e} = p(g|h) * p(e|g,h)

        Args:
            x:                   [B, N, D] token 特征
            z_scores:            [B, G] 逐群 RD z-score
            group_usage:         [B, G] 群使用统计
            group_expert_counts: {group_name: num_experts} 各群专家数

        Returns:
            group_probs:  [B, G] 群概率分布
            expert_probs: {group_name: [B, num_experts]} 各群内专家概率
        """
        # ── Step 1: 构建路由输入 ──
        router_input = self._build_router_input(x, z_scores, group_usage)

        # ── Step 2: 群路由 ──
        group_logits = self.group_router(router_input)  # [B, G]

        # z-score 校正: 高 z-score 的群降低其路由概率
        # score_g = MLP(h)_g - beta * stopgrad(z_g)
        # stopgrad 防止模型通过操纵 z-score 来博弈路由
        if z_scores is not None:
            group_logits = group_logits - self.beta * z_scores.detach()

        group_probs = F.softmax(group_logits / self.tau, dim=-1)  # [B, G]

        # ── Step 3: 专家路由 (每个可扩展群) ──
        expert_probs = {}
        group_names_order = ['Identity', 'SO', 'LR', 'Affine']

        for gn in group_names_order:
            num_exp = (
                group_expert_counts.get(gn, 1)
                if group_expert_counts else 1
            )

            if gn in self.expert_routers and num_exp > 1:
                # 多专家: 使用路由器分配权重
                router = self.expert_routers[gn]
                expert_logits = router(router_input)  # [B, num_exp]
                expert_probs[gn] = F.softmax(expert_logits, dim=-1)
            else:
                # 单专家或不可路由: 均匀权重 (只有一个专家)
                expert_probs[gn] = torch.ones(
                    group_probs.shape[0], 1, device=x.device
                )

        return group_probs, expert_probs


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GroupAwareAE — 群感知自编码器 (逐群表征描述器)
# ═══════════════════════════════════════════════════════════════════════════════

class GroupAwareAE(nn.Module):
    """群感知自编码器 —— 每个群维护独立的 AE 进行重建检测。

    设计 (suggest.md section 8):

    每个群的 AE_g 独立训练:
      L_group_RD = sum_g p(g|h) * MSE(AE_g(z), z)

    核心功能:
      1. 逐群重建: 每个群有独立的 encoder-decoder 对
      2. 群加权损失: 按路由概率加权各群的重建损失
      3. z-score 计算: 基于各群的历史重建误差分布

    群 AE 的回答:
      "当前样本能否被该群条件下的知识解释?"
      重建误差小 -> 可以解释 -> 不需要扩展
      重建误差大 -> 无法解释 -> 触发扩展

    Args:
        dim:        瓶颈维度 r (如 16)
        rd_dim:     AE 压缩维度 (如 4)
        num_groups: 群数量 G
    """

    def __init__(self, dim, rd_dim=None, num_groups=4):
        super().__init__()
        self.dim = dim
        self.rd_dim = rd_dim or max(dim // 4, 4)
        self.num_groups = num_groups

        # ── 逐群 encoder/decoder ──
        # encoder: dim -> rd_dim (压缩到低维)
        self.encoders = nn.ModuleList([
            nn.Linear(dim, self.rd_dim) for _ in range(num_groups)
        ])
        # decoder: rd_dim -> dim (重建回原维度)
        self.decoders = nn.ModuleList([
            nn.Linear(self.rd_dim, dim) for _ in range(num_groups)
        ])

        self._init_weights()

    def _init_weights(self):
        """Kaiming Uniform 初始化所有 encoder/decoder。"""
        for enc, dec in zip(self.encoders, self.decoders):
            nn.init.kaiming_uniform_(enc.weight, a=math.sqrt(5))
            nn.init.zeros_(enc.bias)
            nn.init.kaiming_uniform_(dec.weight, a=math.sqrt(5))
            nn.init.zeros_(dec.bias)

    def forward(self, z, group_idx=None):
        """逐群编码-解码。

        Args:
            z:         [B, dim] 池化后的瓶颈特征
            group_idx: int 或 None (None 返回所有群)

        Returns:
            reconstruction: [B, dim] 或 [G, B, dim]
        """
        if group_idx is not None:
            # 单群重建
            encoded = self.encoders[group_idx](z)
            return self.decoders[group_idx](encoded)
        else:
            # 所有群重建
            reconstructions = []
            for g in range(self.num_groups):
                encoded = self.encoders[g](z)
                reconstructions.append(self.decoders[g](encoded))
            return torch.stack(reconstructions, dim=0)  # [G, B, dim]

    def compute_group_rd_loss(self, z, group_probs):
        """计算群加权重建损失。

        L_group_RD = sum_g p(g|h) * MSE(AE_g(z), z)

        这是扩展检测的核心信号:
        - 每个群的 AE 重建误差反映该群对当前样本的解释能力
        - 按路由概率加权: 模型关注的群获得更高权重

        Args:
            z:           [B, dim] 池化瓶颈特征
            group_probs: [B, G] 群概率

        Returns:
            group_rd_loss:  [B] 逐样本加权重建损失
            per_group_loss: [B, G] 逐群逐样本重建损失
        """
        B, D = z.shape
        G = self.num_groups

        # 所有群重建: [G, B, D]
        all_reconstructions = self.forward(z)

        # 逐群 MSE: [B, G]
        per_group_loss = torch.zeros(B, G, device=z.device)
        for g in range(G):
            per_group_loss[:, g] = F.mse_loss(
                all_reconstructions[g], z, reduction='none'
            ).mean(dim=-1)

        # 群加权求和: [B]
        group_rd_loss = (per_group_loss * group_probs).sum(dim=-1)

        return group_rd_loss, per_group_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RunningRecords — 逐群运行统计缓冲区
# ═══════════════════════════════════════════════════════════════════════════════

class RunningRecords:
    """在线运行统计 —— 逐群维护 RD 重建误差的均值和标准差。

    与原始 Records 类功能相同, 但按群组织。
    每个群的 GroupAwareAE 有对应的 RunningRecords 实例。

    用途:
      Z-score = |当前误差 - 历史均值| / 历史标准差
      Z > exp_threshold -> 分布偏移 -> 触发扩展

    Args:
        max_len: 缓冲区最大容量 (默认 500)
    """

    def __init__(self, max_len=500):
        self._max_len = max_len
        self._curr_len = 0
        self.record = torch.zeros(max_len)
        self._mean = 0.0
        self._var = 0.0
        self.updating = True  # 训练时 True, 冻结后 False

    @property
    def length(self):
        """当前缓冲区中的样本数。"""
        return self._curr_len

    @property
    def mean(self):
        """历史重建误差均值。"""
        return self._mean

    @property
    def stddev(self):
        """历史重建误差标准差 (保证非零)。"""
        return math.sqrt(max(self._var, 1e-8))

    def add_record(self, v):
        """添加新样本并更新运行统计。

        缓冲区管理:
          - 未满: 直接追加
          - 已满: FIFO 滑动窗口, 丢弃最旧样本

        Args:
            v: 标量或 batch 张量 (已 detach 并移至 CPU)
        """
        if not self.updating:
            return
        v = v.detach().cpu()
        if self._curr_len < self._max_len:
            # 缓冲区未满: 直接填充
            place_left = self._max_len - self._curr_len
            if place_left > len(v):
                self.record[self._curr_len:self._curr_len + len(v)] = v
                self._curr_len += len(v)
            else:
                self.record[self._curr_len:] = v[:place_left]
                self._curr_len = self._max_len
        else:
            # 缓冲区已满: FIFO 滑动窗口
            self.record = torch.cat([self.record, v])
            self.record = self.record[len(v):]

        # 更新运行统计
        self._mean = float(torch.mean(self.record[:self._curr_len]))
        self._var = float(torch.var(self.record[:self._curr_len]))


# ═══════════════════════════════════════════════════════════════════════════════
# 8. GroupMoEAdapter — 完整的 Group-MoE 适配器
# ═══════════════════════════════════════════════════════════════════════════════

class GroupMoEAdapter(nn.Module):
    """几何感知 Group-MoE 适配器 —— 替代原始 AdapterModule。

    完整数据流 (suggest.md section 5-8):

      1. z = W_down LN(h)                       -- 瓶颈投影 (共享)
      2. pi = GroupRouter(h, RD_stats)           -- 群 + 专家路由
      3. z^G = sum_g pi_g * T_g(z)              -- Group-MoE 混合
      4. m = SharedMambaFlow(z^G)                -- 共享语义流
      5. a = W_up m                              -- 输出投影
      6. h_out = h + gamma * a                   -- 残差 (gamma 初始为 0)

    同时:
      - GroupAwareAE(z_pooled) -> 逐群 RD 损失 -> z-score
      - z-score 反馈给 Router 进行校正

    Args:
        config:     全局配置 (含 d_model, ffn_num, num_geo_groups 等)
        layer_id:   所在 ViT 层的索引 (0-11)
        adapter_id: 适配器标识符
    """

    def __init__(self, config, layer_id, adapter_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.adapter_id = adapter_id

        d_model = config.d_model                     # ViT 特征维度 (768)
        bottleneck = getattr(config, 'ffn_num', 16)  # 瓶颈维度 r
        num_groups = getattr(config, 'num_geo_groups', 4)  # G = 4 (无 MambaFlow)

        # ── 共享的瓶颈投影 ──
        self.down_proj = nn.Linear(d_model, bottleneck)  # D -> r
        self.up_proj = nn.Linear(bottleneck, d_model)     # r -> D

        # ── 瓶颈前 LayerNorm ──
        self.norm = nn.LayerNorm(d_model)

        # ── 残差门控 (零初始化保证稳定启动) ──
        self.gamma = nn.Parameter(torch.zeros(1))

        # ── GroupBank: 4 个群 (Identity, SO, LR, Affine) ──
        self.group_bank = GroupBank(bottleneck)

        # ── 共享 MambaFlow: 独立于 GroupBank 的语义流算子 ──
        # MambaFlow 不参与群路由混合, 而是在混合后统一应用
        mamba_cfg = {
            'd_state': getattr(config, 'mamba_d_state', 16),
            'd_conv': getattr(config, 'mamba_d_conv', 4),
            'expand': getattr(config, 'mamba_expand', 2),
        }
        self.mamba_flow = SharedMambaFlow(bottleneck, **mamba_cfg)

        # ── 群名称映射 ──
        self.group_names = ['Identity', 'SO', 'LR', 'Affine']
        self.group_name_to_idx = {n: i for i, n in enumerate(self.group_names)}
        self.idx_to_group_name = {i: n for n, i in self.group_name_to_idx.items()}

        # ── 层次路由器 ──
        self.router = HierarchicalRouter(
            d_model, num_groups=num_groups,
            beta=getattr(config, 'router_beta', 0.1),
            tau=getattr(config, 'router_tau', 1.0),
        )
        # 为每个可扩展群初始化专家路由器
        for gn in ['SO', 'LR', 'Affine']:
            self.router.ensure_expert_router(gn, 1)

        # ── 群感知 AE/RD ──
        # 仅在深度层 (9-11) 激活
        self.not_addition_layer = (
            layer_id < config.adapt_start_layer
            or layer_id > config.adapt_end_layer
        )
        if not self.not_addition_layer:
            # GroupAwareAE: 4 个群的独立自编码器
            self.group_ae = GroupAwareAE(
                bottleneck,
                rd_dim=getattr(config, 'rd_dim', 128),
                num_groups=num_groups,
            )
            # 逐群运行统计记录
            self.per_group_records: List[RunningRecords] = [
                RunningRecords(max_len=getattr(config, 'buffer_size', 500))
                for _ in range(num_groups)
            ]
        else:
            self.group_ae = None
            self.per_group_records = None

        self.newly_added = True

        self._init_weights()

    def _init_weights(self):
        """瓶颈投影权重初始化 (遵循原始 SEMA 的 lora 策略)。

        down_proj: Kaiming Uniform — 保证梯度不消失
        up_proj:   全零 — 适配器初始输出为零, 不干扰预训练特征
        """
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)   # 关键: 零初始化, 适配器从零开始
        nn.init.zeros_(self.up_proj.bias)

    def _get_group_expert_counts(self):
        """获取各可扩展群的当前专家数量。"""
        return {
            gn: self.group_bank.num_experts(gn)
            for gn in ['SO', 'LR', 'Affine']
        }

    def _get_group_usage(self):
        """获取群使用统计 (代理: 各群专家数量)。

        Returns:
            usage: [num_groups] 各群使用统计
        """
        usage = torch.zeros(len(self.group_name_to_idx))
        for gn in ['SO', 'LR', 'Affine']:
            idx = self.group_name_to_idx[gn]
            usage[idx] = float(self.group_bank.num_experts(gn))
        return usage

    def _compute_z_scores(self, per_group_loss, detach=True):
        """从逐群 RD 损失计算 z-score。

        z_g = |loss_g - mean_g| / std_g

        高 z-score 表示当前样本在该群下偏离历史分布,
        可能触发扩展。

        Args:
            per_group_loss: [B, G] 逐群重建损失
            detach:         是否 detach (用于 router 校正时需 detach)

        Returns:
            z_scores: [B, G] 逐群 z-score
        """
        B, G = per_group_loss.shape
        z_scores = torch.zeros(B, G, device=per_group_loss.device)
        for g in range(G):
            rec = self.per_group_records[g]
            if rec.length > 2:  # 需要足够的统计样本
                mean = rec.mean
                std = rec.stddev
                loss_g = per_group_loss[:, g].detach() if detach else per_group_loss[:, g]
                z_scores[:, g] = torch.abs((loss_g - mean) / std)
        return z_scores

    def forward(self, x, group_info=None, compute_rd=True):
        """Group-MoE 适配器前向传播。

        完整流程:
          1. 瓶颈投影: z = down_proj(norm(x))
          2. RD z-score 计算 (深度层, 仅当 compute_rd=True)
          3. 层次路由: group_probs, expert_probs
          4. Group-MoE 混合: z^G = sum_g p(g) * T_g(z)
          5. Mamba 语义流: m = MambaFlow(z^G)
          6. 输出投影: a = up_proj(m)
          7. RD 损失计算 (仅当 compute_rd=True)

        func 阶段设 compute_rd=False 跳过 AE (4 encode + 4 decode),
        大幅加速训练。

        Args:
            x:          [B, N, d_model] ViT block 输出 u_l
            group_info: 群位置信息 (可选)
            compute_rd: 是否计算 RD (func 阶段可跳过)

        Returns:
            dict:
              func_out:      [B, N, d_model] 适配器输出 a
              group_rd_loss: scalar 群加权 RD 损失
              z_scores:      [B, G] 逐群 z-score
              group_probs:   [B, G] 群概率
              expert_probs:  {group_name: [B, E]} 专家概率
              added:         bool 是否触发扩展
        """
        B, N, D = x.shape

        # ═══ Step 1: 瓶颈投影 ═══
        z = self.down_proj(self.norm(x))  # [B, N, r]

        # ═══ Step 2: 路由准备 ═══
        group_expert_counts = self._get_group_expert_counts()

        # 深度层: 计算 RD z-score 用于路由器校正 (仅在需要时)
        use_ae = (compute_rd and not self.not_addition_layer
                  and self.group_ae is not None)
        if use_ae:
            z_pooled = z.mean(dim=1)  # [B, r]
            uniform_probs = torch.ones(
                B, len(self.group_names), device=z.device
            ) / len(self.group_names)
            _, per_group_rd = self.group_ae.compute_group_rd_loss(
                z_pooled, uniform_probs
            )
            z_scores = self._compute_z_scores(per_group_rd)
            group_usage = self._get_group_usage().to(z.device).unsqueeze(0).expand(B, -1)
        else:
            z_scores = None
            group_usage = None

        # ═══ Step 3: 层次路由 ═══
        group_probs, expert_probs = self.router(
            x,
            z_scores=z_scores,
            group_usage=group_usage,
            group_expert_counts=group_expert_counts,
        )

        # ═══ Step 4: Group-MoE 混合 ═══
        group_outputs = []
        for i, gn in enumerate(self.group_names):
            g_out, _ = self.group_bank.forward_group(
                gn, z, expert_weights=expert_probs.get(gn)
            )
            group_outputs.append(g_out)

        stacked = torch.stack(group_outputs, dim=0)  # [G, B, N, r]
        w = group_probs.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # [G, B, 1, 1]
        z_G = (stacked * w).sum(dim=0)  # [B, N, r]

        # ═══ Step 5: 共享 MambaFlow ═══
        m = self.mamba_flow(z_G)  # [B, N, r]

        # ═══ Step 6: 输出投影 ═══
        a = self.up_proj(m)  # [B, N, D]

        # ═══ Step 7: RD 损失计算 (仅 compute_rd=True) ═══
        added = False
        group_rd_loss = torch.tensor(0.0, device=x.device)
        z_scores_out = torch.zeros(B, len(self.group_names), device=x.device)

        if use_ae:
            z_pooled = z.mean(dim=1)
            group_rd_loss, per_group_rd = self.group_ae.compute_group_rd_loss(
                z_pooled, group_probs
            )
            group_rd_loss = group_rd_loss.mean()
            if self.training:
                for g in range(len(self.group_names)):
                    self.per_group_records[g].add_record(per_group_rd[:, g])
            z_scores_out = self._compute_z_scores(per_group_rd, detach=False)

        return {
            "func_out": a,
            "group_rd_loss": group_rd_loss,
            "z_scores": z_scores_out,
            "group_probs": group_probs,
            "expert_probs": expert_probs,
            "added": added,
        }

    def add_expert_to_group(self, group_name):
        """向可扩展群添加新专家。

        扩展后需要更新路由器以处理新的专家数量。
        """
        success = self.group_bank.add_expert(group_name)
        if success:
            new_count = self.group_bank.num_experts(group_name)
            self.router.ensure_expert_router(group_name, new_count)
        return success

    def orthogonality_error(self):
        """获取所有 SO 专家的正交误差之和。"""
        return self.group_bank.orthogonality_error()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. GeometrySEMAModules — 层级 Group-MoE 管理器
# ═══════════════════════════════════════════════════════════════════════════════

class GeometrySEMAModules(nn.Module):
    """层级管理器 —— 替代原始 SEMAModules, 管理 Group-MoE 适配器。

    与 SEMAModules 的关键区别:
      - 使用 GroupMoEAdapter 替代标准 Adapter
      - 扩展是群特定专家级别 (而非适配器级别)
      - MambaFlow 共享不扩展
      - 路由器是层次的 (GroupRouter + ExpertRouter)
      - RD 是群感知的 (逐群 AE + 统计)

    扩展检测 (section 9):
      1. 对每个样本, 计算逐群 z-score
      2. 找路由概率最高且 z-score 最高的可扩展群
      3. 若 max_z > exp_threshold: 在该群中添加新专家
      4. 回退: 若所有群 z-score 都高, 优先在最可能的群中扩展

    Args:
        config:   全局配置
        layer_id: ViT 层索引
        writer:   可选的 TensorBoard writer
    """

    def __init__(self, config, layer_id, writer=None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.writer = writer
        self.adapt_start_layer = config.adapt_start_layer
        self.adapt_end_layer = config.adapt_end_layer

        # ── 扩展检测状态 ──
        self.detecting_outlier = False   # 是否处于异常检测模式
        self.added_for_task = False      # 当前任务是否已触发扩展
        self.newly_added = True          # 是否有新添加的组件

        # ── 多 batch 持续性检测 (section 9: "condition persists for multiple batches") ──
        # 累积 z-score 的运行均值, 只有持续超过阈值才触发扩展
        expansion_patience = getattr(config, 'expansion_patience', 3)  # 需要连续 N 个 batch
        self._z_score_accum = torch.zeros(4)  # 累积 z-score (per group)
        self._z_score_count = 0               # 累积 batch 计数
        self._expansion_patience = expansion_patience
        self._expansion_candidate = None      # 候选扩展群 (持续高 z-score 的群)

        # ── 初始化 GroupMoEAdapter ──
        self.adapters: List[GroupMoEAdapter] = nn.ModuleList()
        self.add_adapter(initialize=True)

        # ── 逐群扩展计数 ──
        self.expansion_count = {'SO': 0, 'LR': 0, 'Affine': 0}

    @property
    def num_adapters(self):
        return len(self.adapters)

    def _device(self):
        """获取模块所在设备。"""
        if len(self.adapters) > 0:
            return next(self.adapters[0].parameters()).device
        return torch.device('cpu')

    def add_adapter(self, initialize=False):
        """添加新的 GroupMoEAdapter 实例 (任务 0 初始化或大规模扩展时使用)。"""
        adapter_id = len(self.adapters)
        new_adapter = GroupMoEAdapter(
            self.config, self.layer_id, adapter_id=adapter_id
        ).to(self._device())
        self.newly_added = True
        self.added_for_task = True
        self.adapters.append(new_adapter)
        if not initialize:
            logging.info(
                f"GroupMoEAdapter {self.layer_id}.{adapter_id} 已添加"
            )

    def add_expert_to_group(self, group_name):
        """向当前适配器的指定群添加新专家。

        扩展后精准解冻策略 (防止灾难性遗忘):
          - 新专家参数: 可训练 (新建, 不影响旧知识)
          - 路由器新列: 可训练 (旧列梯度被 hook 清零)
          - 该群的 AE encoder/decoder: 解冻 (只影响被扩展群)
          - 其他群 AE: 保持冻结
          - down_proj/up_proj/MambaFlow: 保持冻结 (共享组件不变)

        持续学习扩展方式 (section 9):
          SO_k -> SO_{k+1}, LR_k -> LR_{k+1}, Affine_k -> Affine_{k+1}
        """
        if self.adapters:
            adapter = self.adapters[-1]
            success = adapter.add_expert_to_group(group_name)
            if success:
                # 1. 解冻路由器新列 (已在 ensure_expert_router 中通过 hook 处理)
                # 2. 精准解冻: 只解冻被扩展群的 AE encoder/decoder
                if adapter.group_ae is not None:
                    group_idx = adapter.group_name_to_idx.get(group_name)
                    if group_idx is not None and group_idx < len(adapter.group_ae.encoders):
                        # 只解冻被扩展群的 encoder
                        for param in adapter.group_ae.encoders[group_idx].parameters():
                            param.requires_grad = True
                        # 只解冻被扩展群的 decoder
                        for param in adapter.group_ae.decoders[group_idx].parameters():
                            param.requires_grad = True
                        logging.info(
                            f"Unfroze group_ae[{group_name}] for RD re-training"
                        )
            return success
        return False

    def forward(self, x, group_info=None):
        """前向传播: Group-MoE + 扩展检测。

        Args:
            x:          [B, N, D] 输入 token 序列
            group_info: 群位置信息 (可选)

        Returns:
            dict:
              func_out:      [B, N, D] 适配器输出
              group_rd_loss: scalar 群 RD 损失
              z_scores:      [B, G] 逐群 z-score
              group_probs:   [B, G] 群概率
              expert_probs:  dict 专家概率
              added:         bool 是否触发扩展
        """
        zero = torch.tensor(0.0, device=x.device)
        not_addition_layer = (
            self.layer_id < self.adapt_start_layer
            or self.layer_id > self.adapt_end_layer
        )

        # ── 浅层/中层: 简单适配器传递, 不检测扩展, 不计算 RD ──
        if not_addition_layer:
            adapter_out = self.adapters[-1](x, group_info=group_info, compute_rd=False)
            return {
                "func_out": adapter_out["func_out"],
                "group_rd_loss": zero,
                "z_scores": adapter_out.get("z_scores"),
                "group_probs": adapter_out.get("group_probs"),
                "expert_probs": adapter_out.get("expert_probs"),
                "added": False,
            }

        # ── 深层 (9-11): 完整 Group-MoE + 扩展检测 ──
        # func 阶段 compute_rd=False 跳过 AE 加速; rd 阶段及检测模式 compute_rd=True
        compute_rd = self.detecting_outlier or getattr(self, '_training_rd', False)
        adapter = self.adapters[-1]
        adapter_out = adapter(x, group_info=group_info, compute_rd=compute_rd)

        # ── 扩展检测逻辑 (section 9): 多 batch 持续性检测 ──
        added = False
        if self.detecting_outlier and not self.added_for_task:
            z_scores = adapter_out.get("z_scores")
            group_probs = adapter_out.get("group_probs")

            if z_scores is not None and group_probs is not None:
                # 更新累积 z-score (运行均值)
                batch_z_mean = z_scores.mean(dim=0).detach().cpu()  # [G]
                self._z_score_accum = (
                    self._z_score_accum * self._z_score_count
                    + batch_z_mean
                ) / (self._z_score_count + 1)
                self._z_score_count += 1

                # 找累积 z-score 最高的可扩展群
                expandable_groups = ['SO', 'LR', 'Affine']
                expandable_indices = [
                    self.adapters[-1].group_name_to_idx[gn]
                    for gn in expandable_groups
                ]
                best_idx = max(expandable_indices,
                              key=lambda i: self._z_score_accum[i].item())
                max_z = self._z_score_accum[best_idx].item()
                best_group = self.adapters[-1].idx_to_group_name[best_idx]

                # 持续性检测: 同一群连续 patience 个 batch 超过阈值才扩展
                if max_z > self.config.exp_threshold:
                    if (self._expansion_candidate == best_group
                            and self._z_score_count >= self._expansion_patience):
                        self.add_expert_to_group(best_group)
                        self.expansion_count[best_group] += 1
                        self.added_for_task = True  # 每任务每层最多扩展 1 次
                        added = True
                        logging.info(
                            f"Block {self.layer_id}: 在群 '{best_group}' 中添加专家 "
                            f"(累积 z={max_z:.3f} > 阈值={self.config.exp_threshold}, "
                            f"持续 {self._z_score_count} batches)"
                        )
                        # 重置累积器
                        self._z_score_accum.zero_()
                        self._z_score_count = 0
                        self._expansion_candidate = None
                    else:
                        # 记录候选群, 等待更多 batch 确认
                        self._expansion_candidate = best_group
                else:
                    # z-score 回落: 清零累积器和候选
                    self._expansion_candidate = None
                    # 不重置累积器 (允许逐步衰减)

        return {
            "func_out": adapter_out["func_out"],
            "group_rd_loss": adapter_out["group_rd_loss"],
            "z_scores": adapter_out.get("z_scores"),
            "group_probs": adapter_out.get("group_probs"),
            "expert_probs": adapter_out.get("expert_probs"),
            "added": added,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # 任务结束管理
    # ═══════════════════════════════════════════════════════════════════════

    def end_of_task_training(self):
        """任务结束时: 冻结所有参数 + 停止 RD 统计更新 + 重置扩展检测状态。"""
        self.freeze_functional()
        self.freeze_rd()
        self.reset_newly_added_status()
        self.added_for_task = False
        # 重置多 batch 累积器
        self._z_score_accum.zero_()
        self._z_score_count = 0
        self._expansion_candidate = None

    def reset_newly_added_status(self):
        """重置 newly_added 标志。"""
        self.newly_added = False
        for adapter in self.adapters:
            adapter.newly_added = False

    def freeze_functional(self):
        """冻结所有功能参数 (投影层, 专家, 路由器)。

        关键: 冻结旧专家防止灾难性遗忘。
        只冻结已有参数, 不影响后续扩展的新专家。
        """
        for adapter in self.adapters:
            # 冻结瓶颈投影
            for param in adapter.down_proj.parameters():
                param.requires_grad = False
            for param in adapter.up_proj.parameters():
                param.requires_grad = False
            # 冻结残差门控
            adapter.gamma.requires_grad_(False)
            # 冻结所有群专家 (旧专家不可训练)
            for gn, experts in adapter.group_bank.groups.items():
                for expert in experts:
                    for param in expert.parameters():
                        param.requires_grad = False
            # 冻结 MambaFlow
            for param in adapter.mamba_flow.parameters():
                param.requires_grad = False
            # 冻结路由器
            for param in adapter.router.parameters():
                param.requires_grad = False

    def freeze_rd(self):
        """冻结 RD 模块: 停止 AE 训练 + 停止统计更新。

        RD 统计反映历史任务分布, 冻结后不再更新,
        以便在新任务中检测分布偏移。
        """
        for adapter in self.adapters:
            if adapter.group_ae is not None:
                for param in adapter.group_ae.parameters():
                    param.requires_grad = False
                if adapter.per_group_records:
                    for rec in adapter.per_group_records:
                        rec.updating = False
