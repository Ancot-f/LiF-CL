"""
Group-Structured Positional Routing (主丛位置路由)
==================================================

实现"群结构位置路由"机制: 在标准 ViT pos_embed 基础上,
维护 K 个可学习的"群位置基"(group positional basis),
通过 Group Router 根据输入特征软选择群结构,
将群路由后的位置信息注入 token 表示。

理论对应:
  - K 个 group_pos_embed 近似主丛 (principal bundle) 上的 K 个局部截面
  - Group Router (group_router) 对应当前输入在选择哪个结构群元素
  - group_scale 控制纤维上的几何扰动幅度
  - 最终 position encoding = base_pos_embed + group_scale * Σ w_k * G_k

Lie 参数版本 (use_lie_param=True):
  不直接学习 group_pos_embed, 而是学习 K 个 2×3 仿射矩阵,
  作用到 patch 坐标上, 再通过 coordinate_mlp 生成位置编码。
  这更接近纤维丛中联络 (connection) 的概念。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupRoutedPositionalEncoding(nn.Module):
    """群结构位置路由: 在标准 pos_embed 之上叠加群路由的位置编码。

    输入:
      x:              [B, N, D]   token 序列 (已加 cls_token, 未加 pos_embed)
      base_pos_embed: [1, N, D]   预训练标准位置编码 (不可训练)
      coords:         [N-1, 2]    可选, patch 二维网格坐标 (不含 cls)

    输出字典:
      x:              [B, N, D]   x + base_pos_embed + group_scale * group_pos
      group_pos:      [B, N, D]   群路由后的位置编码增量
      group_weights:  [B, K]      群权重 (softmax 后)
      group_logits:   [B, K]      群 logits (softmax 前)

    初始化策略:
      - group_pos_embed: trunc_normal_(std=0.02)
      - group_router: 最后一层权重较小, bias=0, 使初始 group_weights 接近均匀
      - group_scale 初始化为小值 (默认 0.1), 避免破坏预训练表示
    """

    def __init__(
        self,
        num_tokens: int,          # N = num_patches + num_tokens (含 cls/dist)
        embed_dim: int,           # D = 768
        num_groups: int = 4,      # K, 群结构数量
        group_scale: float = 0.1, # 初始 scale, 控制群位置的扰动幅度
        use_lie_param: bool = False,
        router_hidden: int = None,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.use_lie_param = use_lie_param

        # 可学习的 group_scale (初始化为小值)
        self.group_scale = nn.Parameter(torch.tensor(group_scale))

        if not use_lie_param:
            # 简单版本: 直接学习 K 个群位置基 [K, N, D]
            self.group_pos_embed = nn.Parameter(
                torch.empty(num_groups, num_tokens, embed_dim)
            )
            nn.init.trunc_normal_(self.group_pos_embed, std=0.02)
        else:
            # Lie 参数版本: 学习 K 个 2×3 仿射矩阵, 作用于 patch 坐标
            # 每个群的参数: 2×3 = 6 维 (2D 仿射变换)
            self.lie_params = nn.Parameter(
                torch.zeros(num_groups, 2, 3)
            )
            # 初始化为单位仿射 (identity)
            with torch.no_grad():
                for k in range(num_groups):
                    self.lie_params[k, 0, 0] = 1.0  # scale_x
                    self.lie_params[k, 1, 1] = 1.0  # scale_y
                    # 其余为零 (translation, shear)
            # 坐标 MLP: 2D → D
            coord_mlp_hidden = router_hidden or embed_dim // 4
            self.coord_mlp = nn.Sequential(
                nn.Linear(2, coord_mlp_hidden),
                nn.GELU(),
                nn.Linear(coord_mlp_hidden, embed_dim),
            )
            # 不可学习的参数, 但可以被训练
            self.group_pos_embed = None  # Lie 模式下动态生成

        # Group Router: 从池化特征选择群结构
        # 使用 2 层 MLP 增强稳定性
        router_hidden_dim = router_hidden or embed_dim // 4
        self.group_router = nn.Sequential(
            nn.Linear(embed_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, num_groups),
        )
        self._init_router()

    def _init_router(self):
        """稳定初始化 group_router: 最后一层小权重 + 零 bias, 初始接近均匀路由."""
        last_layer = self.group_router[-1]  # nn.Linear
        nn.init.trunc_normal_(last_layer.weight, std=0.02)
        nn.init.zeros_(last_layer.bias)
        # 第一层正常初始化
        first_layer = self.group_router[0]  # nn.Linear
        nn.init.trunc_normal_(first_layer.weight, std=0.02)
        nn.init.zeros_(first_layer.bias)

    def _generate_lie_group_pos(self, coords_2d, cls_idx=0):
        """从 Lie 参数动态生成 group_pos_embed.

        Args:
            coords_2d: [N_patches, 2] patch 二维网格坐标 (归一化到 [-1, 1])
            cls_idx: cls token 在序列中的位置索引 (默认 0)

        Returns:
            group_pos: [K, N, D]
        """
        K = self.num_groups
        N_patches = coords_2d.shape[0]
        N = N_patches + 1  # + cls token
        D = self.embed_dim

        # 对每个群, 用 2×3 仿射矩阵变换坐标
        # coords_2d: [N_patches, 2] → [1, N_patches, 2]
        coords = coords_2d.unsqueeze(0).to(self.lie_params.device)  # [1, N_patches, 2]

        # lie_params: [K, 2, 3]
        # 构造齐次坐标: [1, N_patches, 3] (x, y, 1)
        ones = torch.ones(1, N_patches, 1, device=coords.device)
        coords_homo = torch.cat([coords, ones], dim=-1)  # [1, N_patches, 3]

        # 批量仿射变换: [K, 2, 3] × [K, 3, N_patches] → [K, 2, N_patches]
        # bmm 不支持 broadcast, 需要显式 expand
        coords_homo = coords_homo.expand(K, -1, -1)      # [K, N_patches, 3]
        transformed = torch.bmm(
            self.lie_params,                               # [K, 2, 3]
            coords_homo.transpose(1, 2)                    # [K, 3, N_patches]
        )  # [K, 2, N_patches]
        transformed = transformed.transpose(1, 2)  # [K, N_patches, 2]

        # 通过 MLP 生成位置编码
        # [K, N_patches, 2] → [K*N_patches, 2] → MLP → [K*N_patches, D] → [K, N_patches, D]
        flat_coords = transformed.reshape(K * N_patches, 2)
        flat_pos = self.coord_mlp(flat_coords)  # [K*N_patches, D]
        group_pos_patches = flat_pos.reshape(K, N_patches, D)

        # cls token 的群位置: 对 patch 位置取均值
        group_pos_cls = group_pos_patches.mean(dim=1, keepdim=True)  # [K, 1, D]

        # 拼接: [K, cls, N_patches, D] → [K, N, D]
        group_pos = torch.cat([group_pos_cls, group_pos_patches], dim=1)  # [K, N, D]

        return group_pos

    def forward(self, x, base_pos_embed, coords=None):
        """前向传播: 群路由 → 组合 group_pos → 叠加到 token 表示.

        Args:
            x:              [B, N, D]  未加 pos_embed 的 token 序列
            base_pos_embed: [1, N, D]  标准位置编码
            coords:         可选 [N-1, 2] patch 二维坐标 (Lie 模式需要)

        Returns:
            dict: {x, group_pos, group_weights, group_logits}
        """
        B, N, D = x.shape
        K = self.num_groups

        # 1. Group Router: 根据池化特征选择群结构
        pooled = x.mean(dim=1)                      # [B, D]
        group_logits = self.group_router(pooled)    # [B, K]
        group_weights = F.softmax(group_logits, dim=-1)  # [B, K]

        # 2. 获取群位置基
        if self.use_lie_param:
            if coords is None:
                raise ValueError("coords is required when use_lie_param=True")
            group_pos_embed = self._generate_lie_group_pos(coords)  # [K, N, D]
        else:
            group_pos_embed = self.group_pos_embed  # [K, N, D]

        # 3. 用 softmax 权重组合群位置
        # group_weights: [B, K], group_pos_embed: [K, N, D] → group_pos: [B, N, D]
        group_pos = torch.einsum("bk,knd->bnd", group_weights, group_pos_embed)

        # 4. 最终位置编码: 保留原始 + 群扰动
        # base_pos_embed: [1, N, D] → broadcast 到 [B, N, D]
        scale = self.group_scale
        x = x + base_pos_embed + scale * group_pos

        return {
            "x": x,
            "group_pos": group_pos,
            "group_weights": group_weights,
            "group_logits": group_logits,
        }
