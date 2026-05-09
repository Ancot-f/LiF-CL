# ===========================================================================
# 文件名: vit_ssf.py
# 描述: 带 Scale-and-Shift (SSF) 参数高效调优的 Vision Transformer。
#
#   SSF（Scale-and-Shift）是一种参数高效微调方法，在每个线性层和归一化层之后
#   添加可学习的缩放（scale）和偏移（shift）参数。通过仅训练这两个额外参数，
#   SSF 能够在保持预训练权重冻结的前提下实现高效的任务适配。
#
#   公式: y = x * scale + shift
#   其中 scale 和 shift 是可学习的向量，维度与特征维度相同。
#
#   本文件包含:
#     - init_ssf_scale_shift: 初始化 SSF 的 scale 和 shift 参数
#     - ssf_ada: SSF 调制函数，对输入进行逐元素缩放和偏移
#     - Mlp: 带 SSF 调制的 MLP 模块
#     - Attention: 带 SSF 调制的多头自注意力模块
#     - LayerScale: 层缩放模块（LayerScale 正则化）
#     - Block: 带 SSF 调制的标准 Transformer 块
#     - ResPostBlock: 残差后归一化的 Transformer 块
#     - ParallelBlock: 并行的 Transformer 块（多个注意力和 MLP 并行）
#     - PatchEmbed: 带 SSF 调制的 Patch 嵌入层
#     - VisionTransformer: 带 SSF 支持的完整 ViT 主干网络
#     - 各种 ViT 变体的预定义模型构造函数（tiny/small/base/large, SSF 版）
#
#   SSF 参考:
#     - Lian et al., "Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning", NeurIPS 2022
#   ViT 参考:
#     - "An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale"
#     - timm 库: https://github.com/rwightman/pytorch-image-models
# ===========================================================================

import math
import logging
from functools import partial
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv, resolve_pretrained_cfg, checkpoint_seq
from timm.models.layers import DropPath, trunc_normal_, lecun_normal_, _assert
from timm.models.layers.helpers import to_2tuple
from timm.models.registry import register_model


_logger = logging.getLogger(__name__)


# ===========================================================================
# 函数: _cfg
# 描述: 构建模型默认配置字典。设置 ImageNet 默认的归一化均值、标准差、
#       裁剪比例、插值方式等。用于 timm 的模型注册系统。
# ===========================================================================
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


# ===========================================================================
# 字典: default_cfgs
# 描述: 各种 ViT 变体的预定义配置，包括 tiny/small/base/large 的 patch16/32/8
#       以及 ImageNet-1K 和 ImageNet-21K 版本。每个条目包含预训练权重的 URL、
#       输入尺寸、类别数等信息。
# ===========================================================================
default_cfgs = {
    # patch models (weights from official Google JAX impl)
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_base_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch8_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),

    # patch models, imagenet21k (weights from official Google JAX impl)
    'vit_tiny_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        num_classes=21843),
}


# ===========================================================================
# 类: Mlp
# 描述: 带 SSF 调制的 MLP 模块（Vision Transformer 中的前馈网络）。
#       结构: fc1 -> [SSF scale/shift] -> act -> dropout -> fc2 -> [SSF scale/shift] -> dropout
#       当 tuning_mode='ssf' 时，在两个全连接层之后各插入 SSF 调制。
# ===========================================================================
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0., tuning_mode='ssf'):
        """
        参数:
            in_features: 输入特征维度
            hidden_features: 隐藏层维度（默认为 in_features）
            out_features: 输出特征维度（默认为 in_features）
            act_layer: 激活函数类型
            bias: 是否使用偏置（支持对两层分别控制的元组）
            drop: Dropout 概率（支持对两层分别控制的元组）
            tuning_mode: 调优模式，'ssf' 表示启用 Scale-and-Shift
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 支持对两层分别指定 bias 和 dropout
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        # SSF 调制参数初始化
        self.tuning_mode = tuning_mode
        if tuning_mode == 'ssf':
            # scale_1, shift_1: 对 fc1 输出进行调制
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(hidden_features)
            # scale_2, shift_2: 对 fc2 输出进行调制
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(out_features)

    def forward(self, x):
        # 第一层全连接: fc1
        x = self.fc1(x)
        if self.tuning_mode == 'ssf':
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)  # SSF: x * scale + shift

        x = self.act(x)
        x = self.drop1(x)
        # 第二层全连接: fc2
        x = self.fc2(x)
        if self.tuning_mode == 'ssf':
            x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)  # SSF: x * scale + shift

        x = self.drop2(x)

        return x


# ===========================================================================
# 类: Attention
# 描述: 带 SSF 调制的多头自注意力模块。
#       SSF 调制位置: qkv 投影输出后 + 输出投影后
# ===========================================================================
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., tuning_mode='ssf'):
        """
        参数:
            dim: 输入特征维度
            num_heads: 注意力头数
            qkv_bias: Q/K/V 投影是否带偏置
            attn_drop: 注意力权重 dropout 概率
            proj_drop: 输出投影 dropout 概率
            tuning_mode: 调优模式，'ssf' 表示启用 Scale-and-Shift
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # 注意力缩放因子: 1/sqrt(d_k)，防止点积过大
        self.scale = head_dim ** -0.5

        # Q/K/V 合并投影（timm 标准方式，dim -> dim*3）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # SSF 调制参数
        self.tuning_mode = tuning_mode
        if tuning_mode == 'ssf':
            # qkv 输出后的 SSF（维度为 dim*3，因为 Q/K/V 拼接在一起）
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim * 3)
            # 输出投影后的 SSF
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(dim)

    def forward(self, x):
        B, N, C = x.shape
        if self.tuning_mode == 'ssf':
            # qkv 投影 + SSF 调制 + 重塑为多头形式 [3, B, num_heads, N, head_dim]
            qkv = (ssf_ada(self.qkv(x), self.ssf_scale_1, self.ssf_shift_1)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # 拆分 Q, K, V

        # 计算缩放点积注意力: softmax(Q @ K^T / sqrt(d_k)) @ V
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 注意力输出 -> 恢复形状 -> 输出投影
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if self.tuning_mode == 'ssf':
            x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)  # SSF: x * scale + shift
        x = self.proj_drop(x)
        return x


# ===========================================================================
# 类: LayerScale
# 描述: 层缩放模块。
#       对每个特征维度乘以一个可学习的缩放因子 gamma。
#       参考 CaiT 论文: "Going deeper with Image Transformers"
#       公式: y = x * gamma
# ===========================================================================
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        """
        参数:
            dim: 特征维度
            init_values: gamma 的初始值（通常取很小的值如 1e-5）
            inplace: 是否原地操作以节省内存
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# ===========================================================================
# 类: Block
# 描述: 带 SSF 调制的标准 ViT Transformer 块。
#       结构: LN -> [SSF modulation] -> Attention -> DropPath -> + (残差)
#             LN -> [SSF modulation] -> MLP -> DropPath -> + (残差)
#       SSF 调制被插入在每个 LN 的输出之后（Attention/MLP 的输入之前）。
# ===========================================================================
class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, tuning_mode='ssf'):
        """
        参数:
            dim: 特征维度
            num_heads: 注意力头数
            mlp_ratio: MLP 隐藏维度 = dim * mlp_ratio
            qkv_bias: Q/K/V 投影是否带偏置
            drop: MLP 中的 dropout 概率
            attn_drop: 注意力 dropout 概率
            init_values: LayerScale 的初始化值（None 表示不使用 LayerScale）
            drop_path: DropPath（随机深度）概率
            act_layer: 激活函数
            norm_layer: 归一化层类型
            tuning_mode: 调优模式，'ssf' 表示启用 Scale-and-Shift
        """
        super().__init__()
        self.dim = dim
        # 注意力子层: LN -> Attention -> LayerScale -> DropPath
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, tuning_mode=tuning_mode)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP 子层: LN -> MLP -> LayerScale -> DropPath
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop, tuning_mode=tuning_mode)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Block 级别的 SSF 调制参数（应用于 LN 输出）
        self.tuning_mode = tuning_mode
        if tuning_mode == 'ssf':
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim)  # 注意力路径
            self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(dim)  # MLP 路径

    def forward(self, x):
        if self.tuning_mode == 'ssf':
            # SSF 模式: 在 LN 输出后插入 scale/shift 调制
            # 注意力路径: LN -> SSF -> Attention -> LayerScale -> DropPath -> 残差连接
            x = x + self.drop_path1(self.ls1(self.attn(ssf_ada(self.norm1(x), self.ssf_scale_1, self.ssf_shift_1))))
            # MLP 路径: LN -> SSF -> MLP -> LayerScale -> DropPath -> 残差连接
            x = x + self.drop_path2(self.ls2(self.mlp(ssf_ada(self.norm2(x), self.ssf_scale_2, self.ssf_shift_2))))
        else:
            # 标准模式: 无 SSF 调制
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ===========================================================================
# 类: ResPostBlock
# 描述: 残差后归一化的 Transformer 块。
#       与标准 Block 不同，LayerNorm 放在残差连接之后（Post-Norm 架构）。
#       结构: Attention -> LN -> + (残差) -> MLP -> LN -> + (残差)
# ===========================================================================
class ResPostBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        参数: 与 Block 相同
        注意: ResPostBlock 不支持 SSF 调优（不使用 tuning_mode 参数）
        """
        super().__init__()
        self.init_values = init_values

        # Post-Norm 架构: 注意力 -> LayerNorm -> DropPath -> 残差连接
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP -> LayerNorm -> DropPath -> 残差连接
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # 当使用 init_values 时，将 LayerNorm 的 weight 初始化为 init_values
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x):
        # Post-Norm: 先做 Attention/MLP，再做 LN，最后加残差
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


# ===========================================================================
# 类: ParallelBlock
# 描述: 并行 Transformer 块。
#       包含 num_parallel 组注意力和 MLP，它们在同一输入上并行执行，输出求和。
#       可以看作是多个 Transformer 子层的并行集成。
# ===========================================================================
class ParallelBlock(nn.Module):

    def __init__(
            self, dim, num_heads, num_parallel=2, mlp_ratio=4., qkv_bias=False, init_values=None,
            drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        参数:
            dim: 特征维度
            num_heads: 注意力头数
            num_parallel: 并行子层数（默认为 2）
            其余参数同 Block
        """
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        # 创建 num_parallel 组注意力和 MLP 子层
        for _ in range(num_parallel):
            # 注意力子层: LN -> Attention -> LayerScale -> DropPath
            self.attns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('attn', Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))
            # MLP 子层: LN -> MLP -> LayerScale -> DropPath
            self.ffns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('mlp', Mlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))

    def _forward_jit(self, x):
        """JIT 脚本兼容的前向传播（使用 torch.stack）"""
        x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x):
        """标准前向传播（使用 Python sum，更高效）"""
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def forward(self, x):
        """根据运行上下文选择合适的前向传播实现"""
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x)
        else:
            return self._forward(x)


# ===========================================================================
# 类: PatchEmbed
# 描述: 2D 图像到 Patch 序列的嵌入层（带 SSF 调制支持）。
#       使用卷积将图像切分为不重叠的 patch，并嵌入到高维特征空间。
#       支持 SSF 调制应用于卷积输出和最终的 LayerNorm 输出。
# ===========================================================================
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, tuning_mode='ssf'):
        """
        参数:
            img_size: 输入图像尺寸（正方形）
            patch_size: 每个 patch 的尺寸
            in_chans: 输入图像通道数
            embed_dim: 输出嵌入维度
            norm_layer: 可选的归一化层（LayerNorm）
            flatten: 是否将输出的空间维度展平为序列
            tuning_mode: 调优模式，'ssf' 表示启用 Scale-and-Shift
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # patch 网格大小
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.norm_layer = norm_layer

        # 卷积投影: 将图像转换为 patch embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # SSF 调制参数
        self.tuning_mode = tuning_mode
        if tuning_mode == 'ssf':
            # 卷积输出后的 SSF
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(embed_dim)
            if norm_layer:
                # LayerNorm 输出后的 SSF
                self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # 检查输入图像尺寸是否匹配
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")

        # 卷积投影: BCHW -> B x embed_dim x grid[0] x grid[1]
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # 展平空间维度: BCHW -> BNC

        if self.tuning_mode == 'ssf':
            # 卷积输出后的 SSF 调制
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)
            if self.norm_layer:
                # LayerNorm 后的 SSF 调制
                x = ssf_ada(self.norm(x), self.ssf_scale_2, self.ssf_shift_2)
            else:
                x = self.norm(x)
        else:
            x = self.norm(x)
        return x


# ===========================================================================
# 函数: init_ssf_scale_shift
# 描述: 初始化 SSF 的 scale 和 shift 参数。
#       scale 初始化为均值 1 的正态分布（接近恒等映射）
#       shift 初始化为均值 0 的正态分布（接近零偏移）
# ===========================================================================
def init_ssf_scale_shift(dim):
    """初始化 SSF 的 scale（缩放）和 shift（偏移）参数。

    参数:
        dim: 参数的维度（通常等于特征维度）

    返回:
        (scale, shift): 两个可学习的 nn.Parameter
    """
    # scale 从均值 1、标准差 0.02 的正态分布初始化（接近 1，不改变原始输出尺度）
    scale = nn.Parameter(torch.ones(dim))
    # shift 从均值 0、标准差 0.02 的正态分布初始化（接近 0，不产生偏移）
    shift = nn.Parameter(torch.zeros(dim))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift


# ===========================================================================
# 函数: ssf_ada
# 描述: SSF 自适应调制函数。
#       公式: y = x * scale + shift
#       支持两种形状匹配:
#         - 如果 x 的最后一维与 scale 维度相同: 直接逐元素乘加
#         - 如果 x 的第二维（通道维）与 scale 维度相同: 调整 scale/shift 形状为 (1, -1, 1, 1)
#           以适配 BCHW 格式（卷积输出）
# ===========================================================================
def ssf_ada(x, scale, shift):
    """SSF 调制: 对输入进行可学习的缩放和偏移。

    参数:
        x: 输入张量
        scale: 缩放因子向量
        shift: 偏移向量

    返回:
        调制后的张量: x * scale + shift
    """
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        # 序列格式 (BNC): 最后一维是特征维，直接逐元素乘加
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        # 图像格式 (BCHW): 第二维是通道维，需要 broadcast
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')


# ===========================================================================
# 类: VisionTransformer
# 描述: 带 SSF 支持的完整 Vision Transformer 主干网络。
#       支持:
#          - 全局池化（token / avg）
#          - LayerScale 正则化
#          - Post-Norm（ResPostBlock）和并行（ParallelBlock）架构变体
#          - 梯度检查点（gradient checkpointing）以节省显存
#          - 在最终 LN 之后也应用 SSF 调制
# ===========================================================================
class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='',
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, tuning_mode='ssf'):
        """
        Args:
            img_size (int, tuple): 输入图像尺寸
            patch_size (int, tuple): 每个 patch 的尺寸
            in_chans (int): 输入通道数
            num_classes (int): 分类类别数
            global_pool (str): 全局池化方式，可选 '', 'avg', 'token'
            embed_dim (int): 嵌入维度
            depth (int): Transformer 层数
            num_heads (int): 注意力头数
            mlp_ratio (int): MLP 隐藏维度 = embed_dim * mlp_ratio
            qkv_bias (bool): Q/K/V 投影是否带偏置
            init_values: (float): LayerScale 的初始化值
            class_token (bool): 是否使用 CLS token
            fc_norm (Optional[bool]): 全局池化后的 LayerNorm
            drop_rate (float): patch embedding 后的 dropout 概率
            attn_drop_rate (float): 注意力 dropout 概率
            drop_path_rate (float): 随机深度（DropPath）的衰减率
            weight_init (str): 权重初始化方案 ('', 'jax', 'jax_nlhb', 'moco')
            embed_layer (nn.Module): Patch 嵌入层类型
            norm_layer: (nn.Module): 归一化层类型
            act_layer: (nn.Module): MLP 激活层类型
            block_fn: Transformer 块类型 (Block / ResPostBlock / ParallelBlock)
            tuning_mode: 调优模式，'ssf' 表示启用 Scale-and-Shift
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'

        print('Using Pre-trained ViT with Scale & Shift.')
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features 用于与其他模型保持一致性
        self.num_tokens = 1 if class_token else 0
        self.grad_checkpointing = False

        # Patch 嵌入层
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, tuning_mode=tuning_mode)
        num_patches = self.patch_embed.num_patches

        # CLS token（分类 token）: 可学习的参数
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.num_tokens > 0 else None
        # 位置编码: 可学习参数，初始化为正态分布 (std=0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + self.num_tokens, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 计算 DropPath 衰减率（从 0 线性增加到 drop_path_rate）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.tuning_mode = tuning_mode
        tuning_mode_list = [tuning_mode] * depth
        if tuning_mode == 'ssf':
            # 最终 LayerNorm 之后的 SSF 调制参数
            self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(self.num_features)

        # 构建 Transformer 块序列
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, tuning_mode=tuning_mode_list[i])
            for i in range(depth)])

        # 最终的 LayerNorm 和分类头
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        """使用指定的方案初始化模型权重"""
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        """使用 timm 标准方式初始化权重（向下兼容）"""
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        """从 .npz 文件加载预训练权重（Google JAX 格式）"""
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        """返回不参与权重衰减的参数名称集合"""
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        """返回参数分组的正则表达式匹配规则（用于 LR 分组）"""
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem 和 embedding
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """启用/禁用梯度检查点（以计算换内存）"""
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        """获取分类头模块"""
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        """重置分类头为新的类别数（用于增量学习/下游任务）"""
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """
        提取特征（不含分类头）。
        处理流程: Patch Embedding -> 添加 CLS Token -> 位置编码 -> Transformer Blocks -> LayerNorm -> SSF 调制
        """
        x = self.patch_embed(x)
        if self.cls_token is not None:
            # 在序列最前面添加 CLS token
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        # 添加位置编码
        x = self.pos_drop(x + self.pos_embed)

        # 通过 Transformer 编码器层（支持梯度检查点）
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        if self.tuning_mode == 'ssf':
            # 最终 LayerNorm 后的 SSF 调制
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)

        return x

    def forward_head(self, x, pre_logits: bool = False):
        """
        分类头前向传播。
        参数:
            x: 特征张量
            pre_logits: 如果为 True，返回分类头之前的特征（不经过 fc）
        """
        if self.global_pool:
            # 全局池化: avg 或 token
            x = x[:, self.num_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        """完整前向传播: 特征提取 + 分类头"""
        x = self.forward_features(x)
        x = self.forward_head(x)

        return x


# ===========================================================================
# 函数: init_weights_vit_timm
# 描述: ViT 权重初始化（timm 标准方式）。
#       线性层使用截断正态分布（std=0.02），偏置初始化为 0。
#       有自定义 init_weights 方法的模块调用其自身初始化。
# ===========================================================================
def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


# ===========================================================================
# 函数: init_weights_vit_jax
# 描述: ViT 权重初始化（匹配 JAX/Flax 实现）。
#       分类头使用零均值 + 偏置初始化，其他层使用 Xavier 均匀初始化。
#       卷积层使用 Lecun 正态初始化。
# ===========================================================================
def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            # 分类头: weight=0, bias=head_bias
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            # 其他线性层: Xavier 均匀初始化
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                # MLP 中的偏置使用正态分布，其他使用 0
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


# ===========================================================================
# 函数: init_weights_vit_moco
# 描述: ViT 权重初始化（MoCo v3 实现）。
#       qkv 层使用特殊的均匀初始化，其他线性层使用 Xavier 均匀初始化。
# ===========================================================================
def init_weights_vit_moco(module: nn.Module, name: str = ''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # qkv 层: 将 Q, K, V 的权重分开处理
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


# ===========================================================================
# 函数: get_init_weights_vit
# 描述: 根据模式字符串返回对应的初始化函数。
# ===========================================================================
def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


# ===========================================================================
# 函数: _load_weights
# 描述: 从 .npz 文件加载 Google Brain Flax 实现的预训练权重。
#       处理参数名称映射、维度转换、位置编码调整等兼容性问题。
# ===========================================================================
@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        """将 numpy 数组转换为 torch 参数，处理维度排列"""
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            # 转置以匹配 PyTorch 的维度顺序
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    # 加载 patch embedding 权重
    if hasattr(model.patch_embed, 'backbone'):
        # hybrid: 混合模型（CNN + ViT）
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))

    # 加载 CLS token 和位置编码
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # 当尺寸不匹配时调整位置编码
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)

    # 加载最终的 LayerNorm
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))

    # 加载分类头（如果维度匹配）
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))

    # 逐层加载 Transformer 块的权重
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        # 将 JAX 中独立的 Q/K/V kernel 拼接为 qkv.weight
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        # MLP 的两层全连接
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


# ===========================================================================
# 函数: resize_pos_embed
# 描述: 调整位置编码的网格尺寸（从旧分辨率到新分辨率）。
#       使用双三次插值对位置编码的空间部分进行缩放，
#       以适配不同 patch 数量（例如从 224x224 到 384x384）。
# ===========================================================================
def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    """缩放位置编码网格以适配不同的图像尺寸"""
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        # 分离 CLS token 和 patch 位置编码
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # 向后兼容
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    # 重塑为 (1, C, H, W) 进行插值
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    # 恢复为 (1, N, C)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


# ===========================================================================
# 函数: checkpoint_filter_fn
# 描述: 检查点过滤器，用于处理加载权重时的兼容性问题:
#       1. 将旧的展平 patch embedding 权重转换为卷积形式
#       2. 调整位置编码尺寸以匹配当前模型
#       3. 移除 pre_logits（旧版表示层）的权重
# ===========================================================================
def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # 旧版权重: 展平形式 -> 卷积形式
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # 位置编码尺寸不匹配 -> 插值调整
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        elif 'pre_logits' in k:
            # 忽略 pre_logits（最新权重中已移除）
            continue
        out_dict[k] = v
    return out_dict


# ===========================================================================
# 函数: _create_vision_transformer
# 描述: 创建 VisionTransformer 模型的通用工厂函数。
#       使用 timm 的 build_model_with_cfg 处理预训练权重加载和配置解析。
# ===========================================================================
def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model


# ===========================================================================
# 以下为各种 ViT 变体的模型注册和构造函数（使用 @register_model 装饰器）
# 每个函数通过 _create_vision_transformer 创建对应变体并设置正确的参数。
# 名称中的 _ssf 后缀表示这些模型默认启用 SSF 参数高效调优。
# ===========================================================================

@register_model
def vit_tiny_patch16_224_ssf(pretrained=False, **kwargs):
    """ ViT-Tiny (ViT-Ti/16) @ 224x224 """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_tiny_patch16_384_ssf(pretrained=False, **kwargs):
    """ ViT-Tiny (ViT-Ti/16) @ 384x384 """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224_ssf(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16) @ 224x224 (DeiT 论文中的 small 变体) """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_384_ssf(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16) @ 384x384 """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_ssf(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) @ 224x224, ImageNet-1K (from ImageNet-21K fine-tuned) """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_384_ssf(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) @ 384x384 """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224_ssf(pretrained=False, **kwargs):
    """ ViT-Large (ViT-L/16) @ 224x224 """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_384_ssf(pretrained=False, **kwargs):
    """ ViT-Large (ViT-L/16) @ 384x384 """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_tiny_patch16_224_in21k_ssf(pretrained=False, **kwargs):
    """ ViT-Tiny (ViT-Ti/16) @ 224x224, ImageNet-21K 预训练（有 21K 分类头，无表示层） """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224_in21k_ssf(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16) @ 224x224, ImageNet-21K 预训练 """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_in21k_ssf(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) @ 224x224, ImageNet-21K 预训练 """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224_in21k_ssf(pretrained=False, **kwargs):
    """ ViT-Large (ViT-L/16) @ 224x224, ImageNet-21K 预训练 """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model
