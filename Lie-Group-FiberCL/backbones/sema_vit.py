"""
SEMA Vision Transformer 主干网络
================================
基于 ViT-B/16 架构，在每个 Transformer Block 中嵌入 SEMAModules。

与标准 ViT 的关键区别：
  1. 分离的 Q/K/V 投影（而非融合 QKV）— 正确加载标准预训练权重需要
  2. 每个 Block 包含 SEMAModules 用于适配器管理和自扩展检测
  3. 支持 sequential/parallel 两种适配器插入模式 (ffn_option)
  4. 支持 VPT (Visual Prompt Tuning) 作为可选的额外 tuning 方式
  5. forward 返回字典 {"features", "rd_loss", "added_record"}

预训练权重加载：
  - 优先使用 safetensors 格式的预训练权重
  - 自动将融合 QKV 权重拆分为独立的 Q/K/V 投影
  - 自动将 mlp.fc 重命名为 fc（匹配 Block 的命名约定）
  - 加载后冻结所有预训练参数，只训练 adapter/router/new_router

References:
  https://github.com/jxhe/unify-parameter-efficient-tuning
  timm: https://github.com/rwightman/pytorch-image-models
"""
import os
import sys
import timm
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from timm.models.layers import DropPath
from timm.models.registry import register_model
from backbones.sema_block import SEMAModules

try:
    import safetensors.torch
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


class Attention(nn.Module):
    """多头自注意力 —— 使用分离的 Q/K/V 投影（而非融合 QKV）。

    使用分离 Q/K/V 的原因：
        标准 ViT 预训练权重使用融合 QKV 格式（[Q, K, V] 拼接）。
        SEMA 使用分离投影可以更灵活地控制 Q/K/V，
        同时通过 QKV 权重拆分逻辑正确加载预训练权重。
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 分离的 Q/K/V 投影（标准 ViT 使用融合的 nn.Linear(dim, dim*3)）
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """带 SEMA 适配器模块的 ViT Transformer Block。

    数据流：
        x → LayerNorm → Attention → +残差
          → SEMAModules(适配器路由 + 扩展检测) → 得到 adapt_x
          → LayerNorm → MLP → + adapt_x → +残差

    ffn_option 控制适配器插入方式：
        "parallel":  x = MLP(x) + adapter(x)    （默认，适配器与 MLP 并行）
        "sequential": x = adapter(MLP(x))        （适配器串行，在 MLP 之后）
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None, writer=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        if config.ffn_adapt:
            self.adapter_module = SEMAModules(self.config, layer_id=layer_id, writer=writer)
        self.layer_id = layer_id
        self.writer = writer

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        out = self.adapter_module(x)
        adapt_x = out["func_out"]

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if self.config.ffn_adapt:
            if self.config.ffn_option == 'sequential':
                out = self.adapter_module(x)
                x = out["func_out"]
            elif self.config.ffn_option == 'parallel':
                x = x + adapt_x
            else:
                raise ValueError(self.config.ffn_adapt)

        x = residual + x
        out.update({"blk_out": x})
        return out


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None, optim_config=None, writer=None):
        super().__init__()

        print("I'm using ViT with adapters.")
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i, writer=writer
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            self.embeddings = nn.ParameterList(
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in
                 range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

        self.optim_config = optim_config

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    @property
    def feature_dim(self):
        return self.out_dim

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """前向特征提取 —— SEMA 核心数据流。

        遍历所有 SEMA Block，收集：
          - features: 最终 [CLS] token 特征
          - rd_loss: 所有层的表征描述器重建误差之和
          - added_record: 每层是否触发扩展的标志列表

        如果 added=True（某层触发了扩展），提前退出循环。
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        rd_losses = torch.tensor(0., device=x.device)
        added_record = []

        for idx, blk in enumerate(self.blocks):
            if self.tuning_config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            blk_ret = blk(x)
            x = blk_ret["blk_out"]
            rd_loss, added = blk_ret["rd_loss"], blk_ret["added"]
            rd_losses = rd_losses + rd_loss
            added_record.append(added)
            if self.tuning_config.vpt_on:
                x = x[:, self.tuning_config.vpt_num:, :]
            if added:
                break

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        out = {"features": outcome, "rd_loss": rd_losses, "added_record": added_record}
        return out

    def forward(self, x):
        out = self.forward_features(x)
        x = out["features"]
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        out.update({"features": x})
        return out


def _load_pretrained_weights_safetensors(model, pretrained_path):
    """从 safetensors 文件加载预训练 ViT 权重。

    关键处理：
      1. QKV 权重拆分：融合 QKV → 分离的 Q/K/V 投影
      2. MLP 重命名：mlp.fc → fc（匹配 Block 中的 fc1/fc2 命名）
      3. 冻结预训练参数：只保留 adapter/router 参数可训练
    """
    state_dict = safetensors.torch.load_file(pretrained_path)

    # Split fused QKV weights into separate Q, K, V projections
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

    # Rename mlp.fc to fc (matching our Block's fc1, fc2)
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # Freeze all but the adapter parameters
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False

    model.out_dim = 768
    return model


def _load_pretrained_weights_timm(model, timm_model_name):
    """从 timm 加载预训练 ViT 权重（备选方案，当 safetensors 不可用时）。

    同样处理 QKV 权重拆分和 MLP 重命名。
    """
    checkpoint_model = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()

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

    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False

    model.out_dim = 768
    return model


def sema_vit_base_patch16_224(pretrained=True, tuning_config=None, **kwargs):
    """Create SEMA ViT-B/16 model with separated Q/K/V projections.

    This is the standard SEMA backbone:
    - ViT-Base (patch=16, dim=768, depth=12, heads=12)
    - Each Block has SEMAModules for adapter management
    - Separated Q/K/V projections (required for loading fused QKV pretrained weights)
    - Supports sequential/parallel adapter modes (ffn_option)
    - Supports VPT (Visual Prompt Tuning)

    Args:
        pretrained: whether to load pretrained weights
        tuning_config: SEMA tuning configuration
        **kwargs: extra arguments passed to VisionTransformer

    Returns:
        VisionTransformer instance
    """
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), tuning_config=tuning_config, **kwargs
    )

    if pretrained:
        if _HAS_SAFETENSORS:
            # Try lif_cl paths first, then fallback to SEMA-CL-main hardcoded path
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            try:
                from lif_cl.paths import get_premodel_path
                pretrained_path = get_premodel_path("model.safetensors")
            except (ImportError, Exception):
                pretrained_path = "/sdb/syc/My_code/LiF-CL/pre-model/model.safetensors"
            if os.path.exists(pretrained_path):
                model = _load_pretrained_weights_safetensors(model, pretrained_path)
            else:
                print(f"Warning: pretrained weights not found at {pretrained_path}, falling back to timm")
                model = _load_pretrained_weights_timm(model, "vit_base_patch16_224")
        else:
            model = _load_pretrained_weights_timm(model, "vit_base_patch16_224")

    return model


def sema_vit_base_patch16_224_in21k(pretrained=True, tuning_config=None, **kwargs):
    """Create SEMA ViT-B/16 model with ImageNet-21k pretrained weights."""
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), tuning_config=tuning_config, **kwargs
    )

    if pretrained:
        model = _load_pretrained_weights_timm(model, "vit_base_patch16_224_in21k")

    return model
