"""
Flat MoE Vision Transformer Backbone
=====================================

基于 ViT-B/16, 深层 (9-11) 使用 Flat MoE 适配器。

架构:
  Input → PatchEmbed → PosEmbed → Shallow(0-5) → Middle(6-8) → Deep(9-11, Flat MoE) → Pool → Classifier

深层:
  u = Attention(LN(x)) + x
  x = u + MLP(LN(u)) + γ · Σ_e w_e · Expert_e(LN(u))   ← MLP 冻结, 专家可训练

组件:
  - Attention:  Q/K/V 分离多头自注意力
  - Block:      ViT 块 (深层含 Flat MoE)
  - VisionTransformer: 完整主干
  - flat_moe_vit_base_patch16_224: 模型构造函数
"""

import os, sys, timm
from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from timm.models.layers import DropPath
from backbones.flat_moe import ExpertMoEModules
from backbones.sema_geometry import GroupRoutedPositionalEncoding

try:
    import safetensors.torch
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


class Attention(nn.Module):
    """Q/K/V 分离多头自注意力。"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor, seq_len, bsz):
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
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """ViT 块: 深层含 Flat MoE 适配器。

    浅层/中层 (0-8): 标准 ViT (Attn + MLP)
    深层 (9-11):     Attn + MLP(冻结) + Flat MoE(可训练)
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, config=None, layer_id=None, writer=None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm2 = norm_layer(dim)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        self.use_flat_moe = (
            config.ffn_adapt
            and layer_id >= getattr(config, 'adapt_start_layer', 9)
            and layer_id <= getattr(config, 'adapt_end_layer', 11)
        )
        if self.use_flat_moe:
            self.adapter_module = ExpertMoEModules(config, layer_id=layer_id, writer=writer)
        else:
            self.adapter_module = None

    def forward(self, x, group_info=None):
        """前向传播: 返回 dict 含 blk_out, func_out, rd_loss 等。"""
        x = x + self.drop_path(self.attn(self.norm1(x)))

        if self.use_flat_moe:
            # MLP (冻结预训练) + Flat MoE (可训练)
            x_mlp = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
            x_mlp = self.drop_path(self.mlp_drop(self.fc2(x_mlp)))
            u = x
            out = self.adapter_module(u, group_info=group_info)
            adapt_x = out["func_out"]
            x = u + x_mlp + adapt_x
            out.update({"blk_out": x})
            return out
        else:
            residual = x
            x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
            x = self.drop_path(self.mlp_drop(self.fc2(x)))
            x = residual + x
            return {
                "blk_out": x,
                "func_out": torch.zeros_like(x),
                "rd_loss": torch.tensor(0., device=x.device),
                "z_scores": None,
                "expert_weights": None,
                "added": False,
            }


class VisionTransformer(nn.Module):
    """Flat MoE Vision Transformer。

    forward 返回字典, 含 features, logits, rd_loss, added_record 等。
    """

    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, representation_size=None,
                 distilled=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None,
                 optim_config=None, writer=None):
        super().__init__()
        print("I'm using ViT with Flat MoE adapters.")
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                  config=tuning_config, layer_id=i, writer=writer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0
            self.embeddings = nn.ParameterList(
                [nn.Parameter(torch.empty(1, tuning_config.vpt_num, embed_dim))
                 for _ in range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

        self.optim_config = optim_config

        self.use_group_pos = (
            getattr(tuning_config, 'use_group_pos', False)
            if tuning_config is not None else False
        )
        if self.use_group_pos:
            self.group_pos_encoder = GroupRoutedPositionalEncoding(
                num_tokens=num_patches + self.num_tokens, embed_dim=embed_dim,
                num_groups=getattr(tuning_config, 'num_groups', 4),
                group_scale=getattr(tuning_config, 'group_pos_scale', 0.1),
                use_lie_param=getattr(tuning_config, 'use_lie_group_pos', False),
            )
        else:
            self.group_pos_encoder = None

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.use_group_pos and self.group_pos_encoder is not None:
            group_ret = self.group_pos_encoder(x, self.pos_embed)
            x = group_ret["x"]
            group_info = {"group_pos": group_ret["group_pos"],
                          "group_weights": group_ret["group_weights"]}
        else:
            x = x + self.pos_embed
            group_info = None

        x = self.pos_drop(x)

        total_rd_loss = torch.tensor(0., device=x.device)
        added_record = []

        for idx, blk in enumerate(self.blocks):
            if self.tuning_config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)

            blk_ret = blk(x, group_info=group_info)
            x = blk_ret["blk_out"]
            rd = blk_ret.get("rd_loss", torch.tensor(0., device=x.device))
            total_rd_loss = total_rd_loss + (rd if torch.is_tensor(rd) else torch.tensor(0., device=x.device))
            added_record.append(blk_ret.get("added", False))

            if self.tuning_config.vpt_on:
                x = x[:, self.tuning_config.vpt_num:, :]

            if blk_ret.get("added", False):
                break

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return {"features": outcome, "rd_loss": total_rd_loss, "added_record": added_record}

    def forward(self, x):
        out = self.forward_features(x)
        x_feat = out["features"]
        logits = self.head(x_feat)
        out.update({"logits": logits})
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# 权重加载
# ═══════════════════════════════════════════════════════════════════════════════

def _load_pretrained_weights_safetensors(model, pretrained_path):
    state_dict = safetensors.torch.load_file(pretrained_path)
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = qkv_weight[:768]
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = qkv_weight[768:768*2]
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = qkv_weight[768*2:]
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = qkv_bias[:768]
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = qkv_bias[768:768*2]
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = qkv_bias[768*2:]
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            state_dict[key.replace('mlp.', '')] = state_dict.pop(key)
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    for name, p in model.named_parameters():
        p.requires_grad = name in msg.missing_keys
    model.out_dim = 768
    return model


def _load_pretrained_weights_timm(model, timm_model_name):
    checkpoint_model = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = qkv_weight[:768]
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = qkv_weight[768:768*2]
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = qkv_weight[768*2:]
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = qkv_bias[:768]
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = qkv_bias[768:768*2]
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = qkv_bias[768*2:]
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            state_dict[key.replace('mlp.', '')] = state_dict.pop(key)
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    for name, p in model.named_parameters():
        p.requires_grad = name in msg.missing_keys
    model.out_dim = 768
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 模型构造函数
# ═══════════════════════════════════════════════════════════════════════════════

def flat_moe_vit_base_patch16_224(pretrained=True, tuning_config=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tuning_config=tuning_config, **kwargs,
    )
    if pretrained:
        if _HAS_SAFETENSORS:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            try:
                from lif_cl.paths import get_premodel_path
                pretrained_path = get_premodel_path("model.safetensors")
            except (ImportError, Exception):
                pretrained_path = "/sdb/syc/My_code/LiF-CL/pre-model/model.safetensors"
            if os.path.exists(pretrained_path):
                model = _load_pretrained_weights_safetensors(model, pretrained_path)
            else:
                print(f"Warning: pretrained weights not found, falling back to timm")
                model = _load_pretrained_weights_timm(model, "vit_base_patch16_224")
        else:
            model = _load_pretrained_weights_timm(model, "vit_base_patch16_224")
    return model


def flat_moe_vit_base_patch16_224_in21k(pretrained=True, tuning_config=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tuning_config=tuning_config, **kwargs,
    )
    if pretrained:
        model = _load_pretrained_weights_timm(model, "vit_base_patch16_224_in21k")
    return model
