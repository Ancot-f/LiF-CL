"""
李群约束 Adapter 模块 (Lie Group Constrained Adapter)
=====================================================

将 SEMA 标准 Adapter 的 down_proj 约束到 Stiefel 流形上:

    St(768, 16) = { W ∈ R^{768×16} | W^T W = I_16 }

几何意义:
  - W 的列向量是 R^768 中的正交归一基
  - 每个 Adapter 学习一个"视角" — 用 16 维正交基投影特征
  - 不同任务的 Adapter 的基越"远"(测地线距离大) → 分布偏移大

与 SEMA 标准 Adapter 的区别:
  - down_proj.weight 初始化在 Stiefel 上 (标准 Adapter 用 Kaiming Uniform)
  - 每次 optimizer.step() 后投影回 Stiefel (原地 SVD → U @ V^T)
  - 测地线距离替代 Z-score 做扩展检测
  - 其余相同: bottleneck 16, LoRA 风格 up_proj 零初始化, 输出标量缩放
"""

import math
import torch
import torch.nn as nn
from backbones.lie_utils import stiefel_project_, stiefel_init, stiefel_geodesic_distance


class LieAdapter(nn.Module):
    """李群约束的功能适配器 (Stiefel-constrained Bottleneck Adapter)。

    与标准 SEMA Adapter 的核心区别:
        down_proj.weight ∈ St(768, 16) — 约束在 Stiefel 流形上
        而非自由的 R^{768×16}

    约束方式:
        每次 optimizer.step() 后调用 project_() 将 weight 投影回流形。
        这等价于在 Stiefel 流形上做 Riemannian SGD。

    结构:
        x → down_proj (Stiefel) → ReLU → up_proj (free) → output
    """

    def __init__(self, config, adapter_id=None, dropout=0.0, adapter_scalar=1.0):
        """
        Args:
            config: 全局配置 (需含 d_model=768, attn_bn=16)
            adapter_id: 适配器标识符
            dropout: Dropout 率
            adapter_scalar: 输出缩放因子
        """
        super().__init__()
        self.d_model = getattr(config, 'd_model', 768)
        self.bottleneck = getattr(config, 'attn_bn', 16)
        self.adapter_id = adapter_id

        # down_proj: [768, 16], 约束在 Stiefel 上 (W^T W = I_16)
        d, r = self.d_model, self.bottleneck
        self.down_proj = nn.Linear(d, r, bias=False)
        # 初始化为随机 Stiefel 点 (Haar 分布)
        with torch.no_grad():
            self.down_proj.weight.data = stiefel_init(d, r).t()  # nn.Linear stores [r, d]

        self.activation = nn.ReLU()

        # up_proj: [16, 768], 自由参数 (LoRA 风格: 零初始化)
        self.up_proj = nn.Linear(r, d, bias=False)
        with torch.no_grad():
            nn.init.zeros_(self.up_proj.weight)   # 初始输出为零

        self.scalar = float(adapter_scalar)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """前向传播: x → down_proj(Stiefel) → ReLU → up_proj(free)。"""
        h = self.down_proj(x)              # [B,N,768] → [B,N,16]
        h = self.activation(h)
        h = self.dropout(h)
        out = self.up_proj(h)              # [B,N,16] → [B,N,768]
        return out * self.scalar

    def project_(self):
        """在 optimizer.step() 后将 down_proj.weight 投影回 Stiefel 流形。

        down_proj.weight ∈ R^{16×768} (nn.Linear 存储格式)
        约束: weight · weight^T = I_16

        投影: 对 weight [16,768] 做 SVD → U V^T, 使行向量正交归一。

        时间复杂度: O(d·r·min(d,r)) = O(768·16·16) ≈ 200K FLOPs
        """
        with torch.no_grad():
            W = self.down_proj.weight.data  # [16, 768]
            U, _, Vt = torch.linalg.svd(W, full_matrices=False)
            self.down_proj.weight.data = U @ Vt

    def get_stiefel_weight(self):
        """获取 Stiefel 约束的权重矩阵 (列优先格式)。

        Returns:
            W: [768, 16], W^T W = I_16
        """
        return self.down_proj.weight.data.t()  # [16,768] → [768,16]

    def geodesic_distance_to(self, other):
        """计算与另一个 LieAdapter 的测地线距离。

        Args:
            other: LieAdapter 实例

        Returns:
            d: float, Stiefel 流形上两点间的测地线距离
        """
        W1 = self.get_stiefel_weight()       # [768, 16]
        W2 = other.get_stiefel_weight()       # [768, 16]
        return stiefel_geodesic_distance(W1, W2)
