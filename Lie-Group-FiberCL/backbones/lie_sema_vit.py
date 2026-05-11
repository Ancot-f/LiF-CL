"""
李群 SEMA Vision Transformer 主干网络
=====================================

与 sema_vit.py 的区别:
  - Block 使用 LieSEMAModules (而非 SEMAModules)
  - Adapter 的 down_proj 约束在 Stiefel 流形上
  - 扩展检测基于测地线距离 (而非 Z-score)

其余保持一致: 分离 Q/K/V、ffn_option、VPT、QKV 权重拆分。
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
from backbones.lie_sema_block import LieSEMAModules

try:
    import safetensors.torch
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


class Attention(nn.Module):
    """多头自注意力 — 分离 Q/K/V 投影。"""
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
    """带 LieSEMAModules 的 ViT Block。"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None,
                 layer_id=None, writer=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        if config.ffn_adapt:
            self.adapter_module = LieSEMAModules(self.config, layer_id=layer_id, writer=writer)
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
    """带 Lie-SEMA 模块的 Vision Transformer。"""
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
                 weight_init='', tuning_config=None, optim_config=None, writer=None):
        super().__init__()
        print("Using ViT with Lie-SEMA (Stiefel-constrained) adapters.")
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

        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())]))
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
            assert tuning_config.vpt_num > 0
            self.embeddings = nn.ParameterList([
                nn.Parameter(torch.empty(1, tuning_config.vpt_num, embed_dim))
                for _ in range(depth)])
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

    # ────── Stiefel 投影 ──────

    def project_all_adapters_(self):
        """将所有 LieSEMAModules 的 Adapter 投影到 Stiefel 流形。

        在 optimizer.step() 后调用。
        """
        for module in self.modules():
            if isinstance(module, LieSEMAModules):
                module.project_all_()


# ═══════════════════════════════════════════════════════════════
#  模型工厂函数
# ═══════════════════════════════════════════════════════════════

def _load_pretrained_weights(model, pretrained_path):
    """从 safetensors 加载预训练权重并拆分 QKV。"""
    state_dict = safetensors.torch.load_file(pretrained_path)
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = qkv_weight[:768]
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = qkv_weight[768:1536]
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = qkv_weight[1536:]
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = qkv_bias[:768]
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = qkv_bias[768:1536]
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = qkv_bias[1536:]
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            state_dict[key.replace('mlp.', '')] = state_dict.pop(key)

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    for name, p in model.named_parameters():
        p.requires_grad = name in msg.missing_keys
    model.out_dim = 768
    return model


def lie_sema_vit_base_patch16_224(pretrained=True, tuning_config=None, **kwargs):
    """创建 Lie-SEMA ViT-B/16 模型。

    Adapter 的 down_proj 约束在 Stiefel 流形 St(768, 16) 上。
    """
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tuning_config=tuning_config, **kwargs)

    if pretrained and _HAS_SAFETENSORS:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        try:
            from lif_cl.paths import get_premodel_path
            pretrained_path = get_premodel_path("model.safetensors")
        except (ImportError, Exception):
            pretrained_path = "/sdb/syc/My_code/LiF-CL/pre-model/model.safetensors"
        if os.path.exists(pretrained_path):
            model = _load_pretrained_weights(model, pretrained_path)
        else:
            print(f"Pretrained weights not found at {pretrained_path}")

    return model
