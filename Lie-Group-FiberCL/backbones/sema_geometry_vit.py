"""
Geometry-Aware SEMA Vision Transformer Backbone
================================================

基于 ViT-B/16 的几何感知 SEMA 主干网络。
在深层 (9-11) 使用 Group-MoE 适配器, 浅层 (0-5) 和中层 (6-8) 使用标准 ViT。

层级布局 (suggest.md section 3):
  Layers 0-5:  冻结/慢更新 ViT (共享底层视觉与几何基, 不扩展)
  Layers 6-8:  中层语义对齐 (可选轻量几何路由, 高扩展阈值)
  Layers 9-11: 深层持续学习适应层 (Group-MoE 活跃, Group-Aware AE/RD 活跃,
               仅允许群特定专家扩展)

深层输出 (section 10):
  u_l = ViTBlock_l(h_l)            -- ViT 自注意力输出
  a_l = GroupMoEAdapter_l(u_l)     -- Group-MoE 适配器输出
  h_{l+1} = u_l + a_l              -- 残差连接 (gamma 在适配器内部)

组件:
  - Attention:              多头自注意力 (Q/K/V 分离投影)
  - Block:                  ViT 块 (注意力 + Group-MoE/MLP)
  - VisionTransformer:      完整 ViT 主干 (含 Group-MoE 适配器)
  - geo_sema_vit_base_patch16_224:         模型构造函数 (ImageNet-1k 权重)
  - geo_sema_vit_base_patch16_224_in21k:   模型构造函数 (ImageNet-21k 权重)
"""

import os
import sys
import timm
from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from timm.models.layers import DropPath
from backbones.sema_geometry_moe import GeometrySEMAModules
from backbones.sema_geometry import GroupRoutedPositionalEncoding

try:
    import safetensors.torch
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


# ═══════════════════════════════════════════════════════════════════════════════
# Attention — 多头自注意力 (Q/K/V 分离)
# ═══════════════════════════════════════════════════════════════════════════════

class Attention(nn.Module):
    """多头自注意力模块 —— Q/K/V 分离投影。

    与标准 timm ViT Attention 的区别:
      - Q/K/V 使用独立 Linear 层 (而非融合的 qkv Linear)
      - 便于从预训练权重中加载 (预训练通常使用分离投影)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5  # 缩放因子 1/sqrt(d_k)

        # Q/K/V 分离投影
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor, seq_len, bsz):
        """重塑张量为多头格式: [B*num_heads, seq_len, head_dim]。"""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim) \
                     .transpose(1, 2).contiguous()

    def forward(self, x):
        """前向传播: 多头缩放点积注意力。

        Args:
            x: [B, N, C] 输入 token 序列

        Returns:
            x: [B, N, C] 注意力输出
        """
        B, N, C = x.shape

        # 投影 Q/K/V 并重塑为多头格式
        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B) \
                .view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B) \
                .view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B) \
                .view(B * self.num_heads, -1, self.head_dim)

        # 缩放点积注意力
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        # 合并多头 -> 输出投影
        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Block — ViT Transformer 块 (含 Group-MoE 适配器)
# ═══════════════════════════════════════════════════════════════════════════════

class Block(nn.Module):
    """ViT Transformer 块, 深层使用 Geometry-MoE 适配器。

    浅层/中层 (0-8):
      x -> LayerNorm -> Attention -> +residual
      x -> LayerNorm -> MLP (fc1->GELU->fc2) -> +residual
      输出标准 ViT 特征

    深层 (9-11):
      x -> LayerNorm -> Attention -> u_l (残差后)
      u_l -> GeometrySEMAModules -> a_l (Group-MoE + MambaFlow 输出)
      x = u_l + a_l (适配器替代 MLP 位置)
      输出包含路由信息和 RD 损失

    Args:
        dim:        特征维度 (768)
        num_heads:  注意力头数 (12)
        mlp_ratio:  MLP 隐藏层扩展比 (4)
        config:     全局调优配置
        layer_id:   层索引 (0-11)
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, config=None, layer_id=None, writer=None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        # ── 自注意力子层 ──
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # ── MLP 子层 (浅层/中层使用) ──
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm2 = norm_layer(dim)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        # ── Geometry-MoE 适配器 (仅深层) ──
        # 根据 adapt_start_layer / adapt_end_layer 决定是否激活
        self.use_geo_moe = (
            config.ffn_adapt
            and layer_id >= getattr(config, 'adapt_start_layer', 9)
            and layer_id <= getattr(config, 'adapt_end_layer', 11)
        )
        if self.use_geo_moe:
            self.adapter_module = GeometrySEMAModules(
                config, layer_id=layer_id, writer=writer
            )
        else:
            self.adapter_module = None

    def forward(self, x, group_info=None):
        """前向传播。

        Args:
            x:          [B, N, D] 输入 token 序列
            group_info: 群位置路由信息 (可选)

        Returns:
            dict:
              blk_out:       [B, N, D] 块输出
              func_out:      [B, N, D] 适配器输出
              group_rd_loss: scalar 群 RD 损失
              z_scores:      [B, G] 逐群 z-score
              group_probs:   [B, G] 群概率
              expert_probs:  dict 专家概率
              added:         bool 是否触发扩展
        """
        # ── 自注意力 ──
        x = x + self.drop_path(self.attn(self.norm1(x)))

        if self.use_geo_moe:
            # ── 深层: MLP + Group-MoE 适配器 并行 ──
            # MLP 路径 (保留预训练能力, 冻结)
            x_mlp = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
            x_mlp = self.drop_path(self.mlp_drop(self.fc2(x_mlp)))

            # Group-MoE 适配器路径 (可训练, 零初始化不干扰预训练)
            u = x  # 注意力输出作为适配器输入
            out = self.adapter_module(u, group_info=group_info)
            adapt_x = out["func_out"]  # [B, N, D]

            # 残差连接: x = u + MLP(u) + Adapter(u)
            x = u + x_mlp + adapt_x

            # 返回完整输出字典供训练循环使用
            out.update({"blk_out": x})
            return out
        else:
            # ── 浅层/中层: 标准 ViT MLP ──
            residual = x
            x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
            x = self.drop_path(self.mlp_drop(self.fc2(x)))
            x = residual + x

            # 返回与深层一致的字典格式 (含空值)
            return {
                "blk_out": x,
                "func_out": torch.zeros_like(x),
                "group_rd_loss": torch.tensor(0., device=x.device),
                "z_scores": None,
                "group_probs": None,
                "expert_probs": None,
                "added": False,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# VisionTransformer — 完整 Geometry-SEMA ViT 主干
# ═══════════════════════════════════════════════════════════════════════════════

class VisionTransformer(nn.Module):
    """Geometry-SEMA Vision Transformer 主干网络。

    架构 (suggest.md section 3):
      Input -> PatchEmbed -> PosEncoding -> Blocks[0..11] -> Pooling -> Classifier

    其中 Blocks 分布:
      - Blocks 0-5:  浅层 (冻结/慢更新, 无适配器)
      - Blocks 6-8:  中层 (标准 ViT, 可选轻量几何路由)
      - Blocks 9-11: 深层 (Group-MoE + MambaFlow + Group-Aware AE/RD)

    前向返回字典:
      - features:         [B, D] CLS token 特征 (分类用)
      - logits:           [B, num_classes] 分类 logits
      - group_rd_loss:    scalar 累计群 RD 损失
      - added_record:     [12] bool 列表 (每层是否触发扩展)
      - all_group_probs:  list 深层群概率 (用于语义保持)
      - all_z_scores:     list 深层 z-score (用于状态监控)

    Args:
        global_pool:   是否使用全局平均池化 (默认 False, 使用 CLS token)
        img_size:      输入图像尺寸 (224)
        patch_size:    Patch 大小 (16)
        embed_dim:     嵌入维度 (768)
        depth:         Transformer 层数 (12)
        num_heads:     注意力头数 (12)
        tuning_config: Geometry-SEMA 调优配置
    """

    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, representation_size=None,
                 distilled=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None,
                 optim_config=None, writer=None):
        super().__init__()

        print("I'm using ViT with Geometry-MoE adapters.")
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # ── Patch 嵌入 ──
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # ── CLS/Dist Token ──
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)) if distilled else None

        # ── 位置编码 ──
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ── Transformer 块 ──
        # Stochastic Depth (DropPath) 线性递增
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i, writer=writer,
            )
            for i in range(depth)])

        # ── 最终 LayerNorm ──
        self.norm = norm_layer(embed_dim)

        # ── 预 Logits 层 (可选, 用于特征表示学习) ──
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(nn.OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh()),
            ]))
        else:
            self.pre_logits = nn.Identity()

        # ── 分类头 ──
        self.head = nn.Linear(
            self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(
                self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # ── 全局池化选项 ──
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

        # ── VPT (Visual Prompt Tuning) 支持 ──
        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0
            self.embeddings = nn.ParameterList(
                [nn.Parameter(torch.empty(1, tuning_config.vpt_num, embed_dim))
                 for _ in range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

        self.optim_config = optim_config

        # ── Group-Structured Positional Routing (可选) ──
        # 群结构位置路由: 在标准 pos_embed 基础上叠加群路由的位置信息
        self.use_group_pos = (
            getattr(tuning_config, 'use_group_pos', False)
            if tuning_config is not None else False
        )
        if self.use_group_pos:
            self.group_pos_encoder = GroupRoutedPositionalEncoding(
                num_tokens=num_patches + self.num_tokens,
                embed_dim=embed_dim,
                num_groups=getattr(tuning_config, 'num_groups', 4),
                group_scale=getattr(tuning_config, 'group_pos_scale', 0.1),
                use_lie_param=getattr(tuning_config, 'use_lie_group_pos', False),
            )
        else:
            self.group_pos_encoder = None

    def forward_features(self, x):
        """前向特征提取 —— 经过所有 ViT 块并收集深层路由信息。

        遍历 Blocks 0-11, 对深层 (9-11) 收集:
          - group_rd_loss: 群 RD 重建损失
          - added_record: 扩展触发记录
          - all_group_probs: 群路由概率 (用于语义保持)
          - all_z_scores: z-score (用于状态监控)

        Args:
            x: [B, 3, 224, 224] 输入图像

        Returns:
            dict:
              features:         [B, D] CLS token 特征
              group_rd_loss:    scalar 累计 RD 损失
              added_record:     [depth] 每层扩展触发记录
              all_group_probs:  list 深层群概率
              all_z_scores:     list 深层 z-score
        """
        B = x.shape[0]

        # ── Patch 嵌入 ──
        x = self.patch_embed(x)

        # ── 添加 CLS token ──
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # ── 位置编码 ──
        if self.use_group_pos and self.group_pos_encoder is not None:
            # 群结构位置路由
            group_ret = self.group_pos_encoder(x, self.pos_embed)
            x = group_ret["x"]
            group_info = {
                "group_pos": group_ret["group_pos"],
                "group_weights": group_ret["group_weights"],
                "group_logits": group_ret["group_logits"],
            }
        else:
            # 标准位置编码
            x = x + self.pos_embed
            group_info = None

        x = self.pos_drop(x)

        # ── 遍历 Transformer 块 ──
        total_group_rd_loss = torch.tensor(0., device=x.device)
        added_record = []
        all_group_probs = []
        all_z_scores = []

        for idx, blk in enumerate(self.blocks):
            # VPT: 在每层前插入可学习 prompt tokens
            if self.tuning_config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)

            # 前向传播
            blk_ret = blk(x, group_info=group_info)
            x = blk_ret["blk_out"]

            # 收集 RD 损失
            grd = blk_ret.get("group_rd_loss",
                              torch.tensor(0., device=x.device))
            total_group_rd_loss = total_group_rd_loss + (
                grd if torch.is_tensor(grd) else torch.tensor(0., device=x.device)
            )

            # 收集扩展记录
            added = blk_ret.get("added", False)
            added_record.append(added)

            # 收集深层路由信息 (用于语义保持和监控)
            gp = blk_ret.get("group_probs")
            zs = blk_ret.get("z_scores")
            if gp is not None:
                all_group_probs.append(gp.detach())
            if zs is not None:
                all_z_scores.append(zs.detach())

            # VPT: 移除 prompt tokens
            if self.tuning_config.vpt_on:
                x = x[:, self.tuning_config.vpt_num:, :]

            # 扩展触发后跳出 (需要重新训练)
            if added:
                break

        # ── 池化 ──
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # 全局平均池化 (排除 CLS)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]  # CLS token

        return {
            "features": outcome,
            "group_rd_loss": total_group_rd_loss,
            "added_record": added_record,
            "all_group_probs": all_group_probs,
            "all_z_scores": all_z_scores,
        }

    def forward(self, x):
        """前向传播: 特征提取 + 分类。

        Args:
            x: [B, 3, 224, 224] 输入图像

        Returns:
            dict:
              features:         [B, D] CLS token 特征
              logits:           [B, num_classes] 分类 logits
              group_rd_loss:    scalar 累计群 RD 损失
              added_record:     [depth] 扩展触发记录
              all_group_probs:  list 深层群概率
              all_z_scores:     list 深层 z-score
        """
        out = self.forward_features(x)
        x_feat = out["features"]

        # 分类
        if self.head_dist is not None:
            x_main, x_dist = self.head(x_feat[0]), self.head_dist(x_feat[1])
            if self.training and not torch.jit.is_scripting():
                return x_main, x_dist
            else:
                return (x_main + x_dist) / 2
        else:
            logits = self.head(x_feat)

        out.update({"logits": logits})
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# 权重加载辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _load_pretrained_weights_safetensors(model, pretrained_path):
    """从 safetensors 文件加载预训练权重并适配 Q/K/V 分离投影。

    预训练权重通常使用融合的 qkv.weight/qkv.bias,
    需要拆分为独立的 q_proj/k_proj/v_proj。
    同时将 mlp.fc* 映射为 fc1/fc2。

    Args:
        model:           VisionTransformer 实例
        pretrained_path: .safetensors 文件路径

    Returns:
        model: 加载权重后的模型
    """
    state_dict = safetensors.torch.load_file(pretrained_path)

    # 拆分融合的 qkv 权重
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

    # 重命名 MLP 层
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    # 加载权重 (strict=False 允许新增的适配器参数随机初始化)
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # 冻结预训练参数, 只训练新增的适配器参数
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True   # 新参数可训练
        else:
            p.requires_grad = False  # 预训练参数冻结

    model.out_dim = 768
    return model


def _load_pretrained_weights_timm(model, timm_model_name):
    """从 timm 库加载预训练权重。

    与 safetensors 版本相同的处理逻辑: 拆分 Q/K/V, 重命名 MLP。

    Args:
        model:           VisionTransformer 实例
        timm_model_name: timm 模型名称 (如 "vit_base_patch16_224")

    Returns:
        model: 加载权重后的模型
    """
    checkpoint_model = timm.create_model(
        timm_model_name, pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()

    # 拆分融合的 qkv 权重
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

    # 重命名 MLP 层
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # 冻结预训练参数
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False

    model.out_dim = 768
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 模型构造函数
# ═══════════════════════════════════════════════════════════════════════════════

def geo_sema_vit_base_patch16_224(pretrained=True, tuning_config=None, **kwargs):
    """创建 Geometry-SEMA ViT-B/16 模型 (ImageNet-1k 预训练权重)。

    配置:
      - patch_size=16, embed_dim=768, depth=12, num_heads=12
      - 深层 (9-11) 使用 Group-MoE 适配器
      - 加载预训练 ViT 权重, 冻结预训练参数

    Args:
        pretrained:    是否加载预训练权重
        tuning_config: Geometry-SEMA 调优配置 (EasyDict)
        **kwargs:      其他参数 (num_classes, global_pool 等)

    Returns:
        VisionTransformer 实例
    """
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tuning_config=tuning_config, **kwargs,
    )

    if pretrained:
        if _HAS_SAFETENSORS:
            # 尝试从本地 safetensors 加载
            sys.path.insert(0, os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))
            try:
                from lif_cl.paths import get_premodel_path
                pretrained_path = get_premodel_path("model.safetensors")
            except (ImportError, Exception):
                pretrained_path = "/sdb/syc/My_code/LiF-CL/pre-model/model.safetensors"
            if os.path.exists(pretrained_path):
                model = _load_pretrained_weights_safetensors(model, pretrained_path)
            else:
                print(f"Warning: 预训练权重未找到于 {pretrained_path}, "
                      f"回退到 timm")
                model = _load_pretrained_weights_timm(model, "vit_base_patch16_224")
        else:
            model = _load_pretrained_weights_timm(model, "vit_base_patch16_224")

    return model


def geo_sema_vit_base_patch16_224_in21k(pretrained=True, tuning_config=None, **kwargs):
    """创建 Geometry-SEMA ViT-B/16 模型 (ImageNet-21k 预训练权重)。

    与 geo_sema_vit_base_patch16_224 相同的架构,
    但使用 ImageNet-21k 预训练权重以获得更好的特征表示。

    Args:
        pretrained:    是否加载预训练权重
        tuning_config: Geometry-SEMA 调优配置
        **kwargs:      其他参数

    Returns:
        VisionTransformer 实例
    """
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        tuning_config=tuning_config, **kwargs,
    )

    if pretrained:
        model = _load_pretrained_weights_timm(model, "vit_base_patch16_224_in21k")

    return model
