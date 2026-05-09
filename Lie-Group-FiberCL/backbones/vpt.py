"""
Visual Prompt Tuning (VPT) — 视觉提示微调
==========================================

在 ViT 输入空间中插入可学习的 Prompt Token，实现参数高效微调。

两种模式：
  - Shallow VPT: 只在第一层插入 Prompt Token，后续层直接传递
  - Deep VPT: 在每一层都插入独立的 Prompt Token

核心思想：
  冻结整个 ViT 主干，只训练 Prompt Token 和分类头。
  Prompt Token 在 Attention 层中与图像 patch 交互，
  引导模型适应下游任务，同时不改变预训练权重。

关键技巧（Deep VPT）：
  在每层 Attention 前拼接 Prompt Token，
  Attention 计算后立即移除 Prompt Token 对应的输出，
  保证序列长度不变，从而兼容预训练的位置编码。

Reference:
  VPT: Visual Prompt Tuning (ECCV 2022)
  https://arxiv.org/abs/2203.12119
"""

import timm
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, PatchEmbed


def build_promptmodel(modelname='vit_base_patch16_224', Prompt_Token_num=10, VPT_type="Deep"):
    """构建 VPT 模型。

    从 timm 加载预训练 ViT，创建 VPT_ViT 包装器，
    丢弃原始分类头，冻结主干，只保留 Prompt Token 可训练。

    Args:
        modelname: timm 模型名称（'vit_base_patch16_224' 或 'vit_base_patch16_224_in21k'）
        Prompt_Token_num: 每层的 Prompt Token 数量
        VPT_type: "Deep"（每层 Prompt）或 "Shallow"（仅首层 Prompt）

    Returns:
        VPT_ViT: 带 Prompt Token 的 ViT 模型
    """
    # VPT_type = "Deep" / "Shallow"
    edge_size = 224
    patch_size = 16
    num_classes = 1000 if modelname == 'vit_base_patch16_224' else 21843
    basic_model = timm.create_model(modelname, pretrained=True)
    model = VPT_ViT(Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type)

    # 丢弃预训练分类头的权重（CL 场景使用自定义分类头）
    basicmodeldict = basic_model.state_dict()
    basicmodeldict.pop('head.weight')
    basicmodeldict.pop('head.bias')

    model.load_state_dict(basicmodeldict, False)

    # 替换分类头为 Identity（CL 方法自行管理分类头）
    model.head = torch.nn.Identity()

    # 冻结主干，只保留 Prompt Token 可训练
    model.Freeze()

    return model


class VPT_ViT(VisionTransformer):
    """VPT ViT — 在标准 ViT 基础上插入 Prompt Token。

    继承 timm 的 VisionTransformer，重写 forward_features()
    以支持 Deep 和 Shallow 两种 Prompt 插入方式。

    Deep VPT 的关键技巧：
      每层前拼接 Prompt → Attention + MLP 处理 → 移除 Prompt 对应的输出
      这样 Prompt 参与了 Attention 计算但不改变序列长度。
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, Prompt_Token_num=1,
                 VPT_type="Shallow", basic_state_dict=None):

        # 重建标准 ViT 结构
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         norm_layer=norm_layer, act_layer=act_layer)

        print('Using VPT model')
        # 可选：加载基础权重
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        # 根据 VPT 类型创建 Prompt Token
        self.VPT_type = VPT_type
        if VPT_type == "Deep":
            # Deep: 每层独立 Prompt → [depth, Prompt_Token_num, embed_dim]
            self.Prompt_Tokens = nn.Parameter(torch.zeros(depth, Prompt_Token_num, embed_dim))
        else:  # "Shallow"
            # Shallow: 仅首层 Prompt → [1, Prompt_Token_num, embed_dim]
            self.Prompt_Tokens = nn.Parameter(torch.zeros(1, Prompt_Token_num, embed_dim))

    def New_CLS_head(self, new_classes=15):
        """替换分类头（用于增量学习场景）。"""
        self.head = nn.Linear(self.embed_dim, new_classes)

    def Freeze(self):
        """冻结主干网络，只保留 Prompt Token 和分类头可训练。"""
        for param in self.parameters():
            param.requires_grad = False

        self.Prompt_Tokens.requires_grad = True
        try:
            for param in self.head.parameters():
                param.requires_grad = True
        except:
            pass

    def UnFreeze(self):
        """解冻所有参数。"""
        for param in self.parameters():
            param.requires_grad = True

    def obtain_prompt(self):
        """获取 Prompt 状态字典（用于保存/迁移）。"""
        prompt_state_dict = {'head': self.head.state_dict(),
                             'Prompt_Tokens': self.Prompt_Tokens}
        return prompt_state_dict

    def load_prompt(self, prompt_state_dict):
        """加载 Prompt 状态字典（用于恢复/迁移）。"""
        try:
            self.head.load_state_dict(prompt_state_dict['head'], False)
        except:
            print('head not match, so skip head')
        else:
            print('prompt head match')

        if self.Prompt_Tokens.shape == prompt_state_dict['Prompt_Tokens'].shape:
            # 设备检查：确保 Prompt Token 在正确的设备上
            Prompt_Tokens = nn.Parameter(prompt_state_dict['Prompt_Tokens'].cpu())
            Prompt_Tokens.to(torch.device(self.Prompt_Tokens.device))
            self.Prompt_Tokens = Prompt_Tokens
        else:
            print('\n !!! cannot load prompt')
            print('shape of model req prompt', self.Prompt_Tokens.shape)
            print('shape of model given prompt', prompt_state_dict['Prompt_Tokens'].shape)
            print('')

    def forward_features(self, x):
        """带 Prompt Token 的前向特征提取。

        Deep VPT:
          每层前拼接 Prompt Token → Block 处理 → 移除 Prompt 输出
          关键：Prompt 参与了该层的 Attention/MLP 计算，
          但输出时被丢弃，保持序列长度不变。

        Shallow VPT:
          只在第一层前拼接 Prompt Token，后续层正常处理。
        """
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # 拼接 CLS token
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.VPT_type == "Deep":
            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            for i in range(len(self.blocks)):
                # 每层前拼接该层专属的 Prompt Token
                Prompt_Tokens = self.Prompt_Tokens[i].unsqueeze(0)
                x = torch.cat((x, Prompt_Tokens.expand(x.shape[0], -1, -1)), dim=1)
                num_tokens = x.shape[1]
                # Block 处理后移除 Prompt 输出（巧妙技巧）
                x = self.blocks[i](x)[:, :num_tokens - Prompt_Token_num]

        else:  # self.VPT_type == "Shallow"
            Prompt_Token_num = self.Prompt_Tokens.shape[1]

            # 仅首层前拼接 Prompt Token
            Prompt_Tokens = self.Prompt_Tokens.expand(x.shape[0], -1, -1)
            x = torch.cat((x, Prompt_Tokens), dim=1)
            num_tokens = x.shape[1]
            # 所有 Block 顺序处理后移除 Prompt 输出
            x = self.blocks(x)[:, :num_tokens - Prompt_Token_num]

        x = self.norm(x)
        return x

    def forward(self, x):
        """前向传播：特征提取 → 取 [CLS] token 作为输出。"""
        x = self.forward_features(x)
        # 使用 CLS token 作为最终特征
        x = x[:, 0, :]
        return x
