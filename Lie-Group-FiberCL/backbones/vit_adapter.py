"""
AdaptFormer ViT — 带 Adapter 的 Vision Transformer（无 SEMA 扩展的基线方法）
=============================================================================
带 AdaptFormer 风格 Adapter 的 Vision Transformer（无 SEMA 扩展的普通 Adapter）。
这是基于 Adapter 微调的基线骨干网络。

Adapter（适配器）是一种参数高效微调（Parameter-Efficient Tuning）方法，在预训练
模型的 Transformer 层中插入小型瓶颈网络（bottleneck）。Adapter 先降维再升维，
通过少量可训练参数实现对新任务的适配，同时冻结预训练权重不变。

本文件包含:
  - Adapter: 通用的 Adapter 模块（down-proj -> ReLU -> up-proj）
  - Attention: 多头自注意力（Q/K/V 独立投影，用于接收 timm 拆分后的权重）
  - Block: 带 Adapter 的 Transformer 块（支持 Sequential/Parallel 两种插入方式）
  - VisionTransformer: 带 Adapter 和可选的 VPT 的主干 ViT
  - 工厂函数: vit_base_patch16_224_adapter 等，加载预训练权重并冻结非 Adapter 参数

参考文献:
  - AdaptFormer: https://arxiv.org/abs/2205.13535
  - timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
  - DeiT: https://github.com/facebookresearch/deit
  - MAE: https://github.com/facebookresearch/mae
"""

import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import timm
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from timm.models.registry import register_model

import logging
import os
from collections import OrderedDict
import torch


# ===========================================================================
# 类: Adapter
# 描述: 实现 AdaptFormer 风格的 Adapter 模块。
#       结构: LayerNorm(可选) -> Linear(down) -> ReLU -> Dropout -> Linear(up) -> scale -> 残差连接
#       这是一个通用的瓶颈网络，可以插入到 Transformer 的任意位置进行参数高效微调。
# ===========================================================================
class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        """
        参数:
            config: 配置对象，包含 d_model, attn_bn 等字段
            d_model: 输入/输出特征维度（若为 None 则从 config 读取）
            bottleneck: 瓶颈层的降维大小（若为 None 则从 config.attn_bn 读取）
            dropout: Adapter 内部的 dropout 概率
            init_option: 初始化方式，'bert' 或 'lora'（lora 使用 Kaiming 初始化）
            adapter_scalar: 输出缩放因子，可以是浮点数或 "learnable_scalar"（可学习）
            adapter_layernorm_option: LayerNorm 位置，'in'（前面），'out'（后面），或 None
        """
        super().__init__()
        # 确定 Adapter 的输入维度和瓶颈维度
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        # 记录 LayerNorm 放置策略（在 Adapter 之前还是之后）
        self.adapter_layernorm_option = adapter_layernorm_option

        # 若需要，在 Adapter 内部创建 LayerNorm
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        # 确定缩放因子: 可以是固定值或可学习的参数
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        # 降维投影（down-projection）：d_model -> bottleneck
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        # 非线性激活函数
        self.non_linear_func = nn.ReLU()
        # 升维投影（up-projection）：bottleneck -> d_model
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        # 初始化 Adapter 参数
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            # LoRA 风格的初始化: 下投影用 Kaiming 均匀初始化，上投影初始化为零
            # 这样 Adapter 初始时近似恒等变换，不影响预训练模型输出
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        """
        前向传播。
        参数:
            x: 输入特征张量
            add_residual: 是否添加残差连接
            residual: 残差项，若为 None 则使用 x 本身作为残差
        返回:
            输出特征张量
        """
        # 如果未指定残差，默认使用输入 x 作为残差
        residual = x if residual is None else residual

        # 如果 LayerNorm 位于 Adapter 之前，先做归一化
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        # 降维 -> 非线性激活 -> Dropout -> 升维
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        # 对输出进行缩放
        up = up * self.scale

        # 如果 LayerNorm 位于 Adapter 之后，再做归一化
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        # 残差连接
        if add_residual:
            output = up + residual
        else:
            output = up

        return output


# ===========================================================================
# 类: Attention
# 描述: 多头自注意力模块。
#       与标准 ViT 注意力不同，这里将 Q/K/V 投影拆分为独立的 Linear 层
#       （q_proj, k_proj, v_proj），以便与 timm 预训练权重中的 qkv.weight
#       拆分后的权重兼容。
# ===========================================================================
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        """
        参数:
            dim: 输入特征维度
            num_heads: 注意力头数
            qkv_bias: Q/K/V 投影是否带偏置
            attn_drop: 注意力权重上的 dropout 概率
            proj_drop: 输出投影上的 dropout 概率
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        # 缩放因子: 1/sqrt(d_k)，用于防止点积过大导致 softmax 梯度消失
        self.scale = head_dim ** -0.5

        # Q/K/V 独立投影（与 timm 中使用独立 q_proj/k_proj/v_proj 的变体兼容）
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """将张量重塑为多头形式: (B, N, C) -> (B, num_heads, N, head_dim)"""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        # 分别计算 Q, K, V
        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        # 批量矩阵乘法计算注意力权重
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        # Softmax 归一化 + Dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        # 注意力输出 = 注意力权重 x Value
        attn_output = torch.bmm(attn_probs, v)

        # 恢复原始形状: (B*num_heads, N, head_dim) -> (B, N, C)
        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        # 输出投影 + Dropout
        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


# ===========================================================================
# 类: Block
# 描述: 带 Adapter 的 Transformer 块。
#       标准结构: LN -> Attention -> + (残差) -> LN -> MLP -> + (残差)
#       Adapter 可以以两种方式插入到 MLP 子层中:
#         - 'sequential' (串行): 在 MLP 之后顺序执行 Adapter
#         - 'parallel'   (并行): Adapter 与 MLP 并行处理输入，输出相加
# ===========================================================================
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        """
        参数:
            dim: 特征维度
            num_heads: 注意力头数
            mlp_ratio: MLP 隐藏层维度与输入维度的比例
            qkv_bias: Q/K/V 投影是否带偏置
            drop: 全连接层的 dropout 概率
            attn_drop: 注意力 dropout 概率
            drop_path: DropPath（随机深度）的概率
            act_layer: 激活函数类型
            norm_layer: 归一化层类型
            config: 调优配置对象，控制 Adapter 的行为
            layer_id: 当前层的索引（从 0 开始）
        """
        super().__init__()
        self.config = config
        # 第一个 LayerNorm + 自注意力
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # DropPath 用于随机深度（Stochastic Depth），训练时随机丢弃整个块
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 第二个 LayerNorm + MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # 标准的两层全连接 MLP: fc1 -> act -> dropout -> fc2
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        # 如果配置启用 FFN Adapter，创建 Adapter 模块（插入到 MLP 子层中）
        if config.ffn_adapt:
            self.adaptmlp = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                                    init_option=config.ffn_adapter_init_option,
                                    adapter_scalar=config.ffn_adapter_scalar,
                                    adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                    )

    def forward(self, x):
        # 注意力子层：LN -> Attention -> DropPath -> 残差连接
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # 如果使用并行模式的 Adapter，先计算 Adapter 输出（不加重残差）
        if self.config.ffn_adapt and self.config.ffn_option == 'parallel':
            adapt_x = self.adaptmlp(x, add_residual=False)

        # MLP 子层：LN -> fc1 -> act -> dropout -> fc2 -> dropout -> DropPath
        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        # 根据配置选择 Adapter 与 MLP 的组合方式
        if self.config.ffn_adapt:
            if self.config.ffn_option == 'sequential':
                # 串行模式: MLP 输出后接 Adapter
                x = self.adaptmlp(x)
            elif self.config.ffn_option == 'parallel':
                # 并行模式: Adapter 与 MLP 的输出相加
                x = x + adapt_x
            else:
                raise ValueError(self.config.ffn_adapt)

        # MLP 子层的残差连接
        x = residual + x
        return x


# ===========================================================================
# 类: VisionTransformer
# 描述: 带 Adapter 和可选 VPT（Visual Prompt Tuning）的 Vision Transformer 主干网络。
#       支持:
#          - 全局平均池化（global_pool）
#          - Adapter 微调（通过 Block 中的 adaptmlp）
#          - VPT（Visual Prompt Tuning）: 在每层输入前插入可学习的 prompt token
#          - 蒸馏 token（dist_token）
# ===========================================================================
class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None):
        """
        参数:
            global_pool: 是否使用全局平均池化（替代 CLS token）
            img_size: 输入图像尺寸
            patch_size: 每个图像块的尺寸（ViT 将图像切分成不重叠的块）
            in_chans: 输入图像的通道数
            num_classes: 分类类别数
            embed_dim: 词嵌入维度（Token Embedding 维度）
            depth: Transformer 编码器的层数
            num_heads: 注意力头数
            mlp_ratio: MLP 隐藏层维度与嵌入维度的比例
            qkv_bias: Q/K/V 投影是否带偏置
            representation_size: 预分类表示的维度（None 表示不使用）
            distilled: 是否包含蒸馏 token（DeiT 风格）
            drop_rate: patch embedding 层的 dropout 概率
            attn_drop_rate: 注意力 dropout 概率
            drop_path_rate: DropPath 的概率（用于随机深度）
            embed_layer: Patch 嵌入层类型
            norm_layer: 归一化层类型
            act_layer: 激活函数类型
            weight_init: 权重初始化方式
            tuning_config: 微调配置，控制 Adapter 和 VPT 的行为
        """
        super().__init__()


        print("I'm using ViT with adapters.")
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features 用于与其他模型保持一致性
        self.num_tokens = 2 if distilled else 1  # 蒸馏模型有额外的 dist_token
        # 默认使用 LayerNorm 和 GELU
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # Patch 嵌入: 将图像切分为不重叠的 patch，并通过卷积编码为 token
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token: 用于图像分类的分类 token，可学习
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 蒸馏 token（DeiT 风格），可选
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # 位置编码: 可学习的参数，为每个 patch + CLS token 提供位置信息
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 计算随机深度（DropPath）的衰减率，从 0 线性增加到 drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # 构建 Transformer 编码器层
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)])
        # 最终 LayerNorm
        self.norm = norm_layer(embed_dim)

        # 表示层（预分类头）: 可选的线性层 + Tanh 激活，用于生成中间表示
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # 分类头: 将最终的表示映射到类别对数几率（logits）
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            # 蒸馏模型有独立的分类头用于蒸馏 token
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        ######### MAE 风格的全局池化设置 ############
        self.global_pool = global_pool
        if self.global_pool:
            # 全局平均池化模式: 使用 fc_norm 替代原始的 norm
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # 移除原始的 norm

        ######## VPT (Visual Prompt Tuning) 设置 #########
        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            # 为每一层创建可学习的 prompt token（Deep VPT 模式）
            # 形状: [depth, num_prompt, embed_dim]
            self.embeddings = nn.ParameterList(
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in
                 range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        """返回不参与权重衰减的参数集合（pos_embed, cls_token, dist_token）"""
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        """获取分类头（支持蒸馏模式返回两个头）"""
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        """重置分类头为新的类别数（用于增量学习场景）"""
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """
        提取特征（不含分类头）。
        处理流程: Patch Embedding -> 添加 CLS Token -> 位置编码 -> Transformer Blocks -> LayerNorm
        支持 VPT (Visual Prompt Tuning):
          - Deep VPT: 在每一层输入前添加可学习的 prompt tokens，该层输出后移除
          - Shallow VPT: 只在第一层输入前添加 prompt tokens
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # 将图像转换为 patch token 序列

        # 添加 CLS token（分类 token）到序列的最前面
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 逐层通过 Transformer 编码器块
        for idx, blk in enumerate(self.blocks):
            if self.tuning_config.vpt_on:
                # Deep VPT: 在当前层输入前拼接 prompt tokens
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            x = blk(x)
            if self.tuning_config.vpt_on:
                # 移除 prompt tokens，仅保留原始序列用于下一层
                x = x[:, self.tuning_config.vpt_num:, :]

        # 最终归一化和池化
        if self.global_pool:
            # 全局平均池化: 对除 CLS token 外的所有 token 取均值
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            # 标准模式: 取 CLS token 对应的输出
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        """完整前向传播: 特征提取 + 分类头"""
        x = self.forward_features(x,)
        if self.head_dist is not None:
            # 蒸馏模式: CLS token 和 dist token 分别通过不同的分类头
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # 训练时返回两个分类头的输出
                return x, x_dist
            else:
                # 推理时取两个分类头的平均值
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


# ===========================================================================
# 函数: vit_base_patch16_224_adapter
# 描述: 创建 ViT-B/16 @ 224x224 的 Adapter 模型。
#       加载 timm 预训练权重（ImageNet-1K），将 qkv 权重拆分为独立的 q/k/v 权重，
#       将 mlp.fc 重命名为 fc，然后冻结除 Adapter 外的所有参数。
# ===========================================================================
def vit_base_patch16_224_adapter(pretrained=False, **kwargs):
    """创建带 Adapter 的 ViT-B/16（224x224 分辨率，ImageNet-1K 预训练）"""

    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # 使用 timm 加载预训练的 ViT-B/16 权重
    checkpoint_model=timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    # 修改 checkpoint 的 state_dict 以匹配模型的命名约定
    # 第一步: 将 qkv.weight 拆分为 q_proj.weight, k_proj.weight, v_proj.weight
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]         # 前 768 维是 Q
            k_weight = qkv_weight[768:768*2]    # 中间 768 维是 K
            v_weight = qkv_weight[768*2:]       # 后 768 维是 V
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            # 同样地拆分 qkv.bias
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # 第二步: 将 mlp.fc.weight 重命名为 fc.weight（去掉 mlp. 前缀）
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    # 加载权重（strict=False 忽略 Adapter 等新增参数）
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # 冻结除 Adapter 外的所有参数，只训练 Adapter 参数
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True   # Adapter 参数（checkpoint 中不存在的键）
        else:
            p.requires_grad = False  # 预训练参数（冻结）
    return model


# ===========================================================================
# 函数: vit_base_patch16_224_in21k_adapter
# 描述: 创建 ViT-B/16 @ 224x224 的 Adapter 模型。
#       与上面的函数类似，但加载 ImageNet-21K 预训练权重而非 ImageNet-1K。
#       同样拆分 qkv 权重、重命名 mlp.fc，并冻结除 Adapter 外的所有参数。
# ===========================================================================
def vit_base_patch16_224_in21k_adapter(pretrained=False, **kwargs):
    """创建带 Adapter 的 ViT-B/16（224x224 分辨率，ImageNet-21K 预训练）"""

    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # 使用 timm 加载 ImageNet-21K 预训练的 ViT-B/16 权重
    checkpoint_model=timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    # 第一步: 将 qkv.weight / qkv.bias 拆分为独立的 q_proj, k_proj, v_proj
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # 第二步: 将 mlp.fc 重命名为 fc
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    # 加载权重
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # 冻结预训练参数，仅训练 Adapter 和新增参数
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model
