"""
增量学习网络包装器
=================
提供所有持续学习方法使用的网络包装类和主干网络工厂函数。

网络包装器：
  - BaseNet: 基础包装器（backbone + fc 分类头）
  - SEMAVitNet: SEMA 专用包装器，处理 backbone 的字典输出
  - IncrementalNet: 标准增量学习网络
  - PromptVitNet: L2P/DualPrompt 的 Prompt 网络
  - CodaPromptVitNet: CODA-Prompt 网络
  - FOSTERNet: FOSTER 的特征增强网络
  - DERNet / AdaptiveNet: 其他方法专用网络

主干网络工厂 (get_backbone):
  根据 backbone_type 参数动态创建对应的 ViT 变体：
  - pretrained_vit_b16_224_adapter → SEMA ViT
  - pretrained_vit_b16_224_ssf → SSF ViT
  - pretrained_vit_b16_224_l2p → L2P ViT
  - ... 等
"""

import copy
import logging
import torch
from torch import nn
from backbones.linears import SimpleLinear, SplitCosineLinear, CosineLinear, SimpleContinualLinear
from backbones.prompt import CodaPrompt
import timm


def get_backbone(args, pretrained=False):
    """主干网络工厂函数 —— 根据 backbone_type 创建对应的 ViT 变体。

    支持的 backbone_type 命名规则：
      - '_adapter' 结尾 → SEMA / AdaptFormer ViT
      - '_ssf' 结尾 → Scale-and-Shift ViT
      - '_vpt' 结尾 → Visual Prompt Tuning ViT
      - '_l2p' 结尾 → Learning to Prompt ViT
      - '_dualprompt' 结尾 → DualPrompt ViT
      - '_coda_prompt' 结尾 → CODA-Prompt ViT
      - '_memo' 结尾 → MEMO ViT
      - 无后缀 → 标准预训练 ViT

    Args:
        args: 参数字典（需含 backbone_type, model_name 等）
        pretrained: 是否加载预训练权重

    Returns:
        nn.Module: 对应类型的 ViT 模型
    """
    name = args["backbone_type"].lower()
    # SimpleCIL or SimpleCIL w/ Finetune
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()

    elif '_memo' in name:
        if args["model_name"] == "memo":
            from backbones import vit_memo
            _basenet, _adaptive_net = timm.create_model("vit_base_patch16_224_memo", pretrained=True, num_classes=0)
            _basenet.out_dim = 768
            _adaptive_net.out_dim = 768
            return _basenet, _adaptive_net
    # SSF
    elif '_ssf' in name:
        if args["model_name"] == "aper_ssf":
            from backbones import vit_ssf
            if name == "pretrained_vit_b16_224_ssf":
                model = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
                model.out_dim = 768
            elif name == "pretrained_vit_b16_224_in21k_ssf":
                model = timm.create_model("vit_base_patch16_224_in21k_ssf", pretrained=True, num_classes=0)
                model.out_dim = 768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    # VPT
    elif '_vpt' in name:
        if args["model_name"] == "aper_vpt":
            from backbones.vpt import build_promptmodel
            if name == "pretrained_vit_b16_224_vpt":
                basicmodelname = "vit_base_patch16_224"
            elif name == "pretrained_vit_b16_224_in21k_vpt":
                basicmodelname = "vit_base_patch16_224_in21k"

            print("modelname,", name, "basicmodelname", basicmodelname)
            VPT_type = "Deep"
            if args["vpt_type"] == 'shallow':
                VPT_type = "Shallow"
            Prompt_Token_num = args["prompt_token_num"]

            model = build_promptmodel(modelname=basicmodelname, Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type)
            prompt_state_dict = model.obtain_prompt()
            model.load_prompt(prompt_state_dict)
            model.out_dim = 768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    elif '_adapter' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "lie_sema":
            from backbones import lie_sema_vit
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer + Stiefel constraint
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                ffn_adapter_type=args["ffn_adapter_type"],
                d_model=768,
                attn_bn=ffn_num,
                # VPT
                vpt_on=False,
                vpt_num=0,
                # Lie-SEMA
                exp_threshold=args["exp_threshold"],
                geo_threshold=args.get("geo_threshold", 0.5),
                adapt_start_layer=args["adapt_start_layer"],
                adapt_end_layer=args["adapt_end_layer"],
                rd_dim=args["rd_dim"],
                buffer_size=args["buffer_size"],
            )
            if name == "pretrained_vit_b16_224_adapter":
                model = lie_sema_vit.lie_sema_vit_base_patch16_224(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim = 768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = lie_sema_vit.lie_sema_vit_base_patch16_224(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim = 768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        elif args["model_name"] == "geo_sema":
            from backbones import sema_geometry_vit
            from easydict import EasyDict
            tuning_config = EasyDict(
                # Geometry-MoE
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                ffn_adapter_type=args["ffn_adapter_type"],
                d_model=768,
                attn_bn=ffn_num,
                # VPT
                vpt_on=False,
                vpt_num=0,
                # Geometry-MoE specific
                exp_threshold=args["exp_threshold"],
                adapt_start_layer=args["adapt_start_layer"],
                adapt_end_layer=args["adapt_end_layer"],
                rd_dim=args["rd_dim"],
                buffer_size=args["buffer_size"],
                # 多 batch 持续性检测
                expansion_patience=args.get("expansion_patience", 3),
                # Group-MoE
                num_geo_groups=args.get("num_geo_groups", 4),  # Identity, SO, LR, Affine (MambaFlow 独立)
                router_beta=args.get("router_beta", 0.1),
                router_tau=args.get("router_tau", 1.0),
                # MambaFlow
                mamba_d_state=args.get("mamba_d_state", 16),
                mamba_d_conv=args.get("mamba_d_conv", 4),
                mamba_expand=args.get("mamba_expand", 2),
                # Group-Structured Positional Routing
                use_group_pos=args.get("use_group_pos", False),
                num_groups=args.get("num_groups", 4),
                group_pos_scale=args.get("group_pos_scale", 0.1),
                use_lie_group_pos=args.get("use_lie_group_pos", False),
            )
            if name == "pretrained_vit_b16_224_adapter":
                model = sema_geometry_vit.geo_sema_vit_base_patch16_224(
                    num_classes=0, global_pool=False, drop_path_rate=0.0,
                    tuning_config=tuning_config)
                model.out_dim = 768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = sema_geometry_vit.geo_sema_vit_base_patch16_224_in21k(
                    num_classes=0, global_pool=False, drop_path_rate=0.0,
                    tuning_config=tuning_config)
                model.out_dim = 768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        elif args["model_name"] == "group_basis_moe":
            from backbones import group_basis_vit
            from easydict import EasyDict
            tuning_config = EasyDict(
                ffn_adapt=True, ffn_option="parallel", ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora", ffn_adapter_scalar="0.1",
                ffn_num=ffn_num, ffn_adapter_type=args["ffn_adapter_type"],
                d_model=768, attn_bn=ffn_num, vpt_on=False, vpt_num=0,
                init_bases=args.get("init_bases", 2),
                exp_threshold=args["exp_threshold"],
                expansion_patience=args.get("expansion_patience", 3),
                adapt_start_layer=args["adapt_start_layer"],
                adapt_end_layer=args["adapt_end_layer"],
                rd_dim=args.get("rd_dim", 32), buffer_size=args["buffer_size"],
                num_groups=args.get("num_groups", 4),
                router_beta=args.get("router_beta", 0.1),
                router_tau=args.get("router_tau", 1.0),
                protect_threshold=args.get("protect_threshold", 2.0),
                use_group_pos=args.get("use_group_pos", False),
                num_groups_pos=args.get("num_groups", 4),
                group_pos_scale=args.get("group_pos_scale", 0.1),
                use_lie_group_pos=args.get("use_lie_group_pos", False),
            )
            if name == "pretrained_vit_b16_224_adapter":
                model = group_basis_vit.group_basis_vit_base_patch16_224(
                    num_classes=0, global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim = 768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = group_basis_vit.group_basis_vit_base_patch16_224(
                    num_classes=0, global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim = 768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        elif args["model_name"] == "flat_moe":
            from backbones import flat_moe_vit
            from easydict import EasyDict
            tuning_config = EasyDict(
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                ffn_adapter_type=args["ffn_adapter_type"],
                d_model=768,
                attn_bn=ffn_num,
                vpt_on=False,
                vpt_num=0,
                # Flat MoE
                init_experts=args.get("init_experts", 1),
                exp_threshold=args["exp_threshold"],
                expansion_patience=args.get("expansion_patience", 3),
                adapt_start_layer=args["adapt_start_layer"],
                adapt_end_layer=args["adapt_end_layer"],
                rd_dim=args.get("rd_dim", 64),
                buffer_size=args["buffer_size"],
                router_beta=args.get("router_beta", 0.1),
                router_tau=args.get("router_tau", 1.0),
                use_so_reg=args.get("use_so_reg", True),
                use_lr_reg=args.get("use_lr_reg", False),
                use_group_pos=args.get("use_group_pos", False),
                num_groups=args.get("num_groups", 4),
                group_pos_scale=args.get("group_pos_scale", 0.1),
                use_lie_group_pos=args.get("use_lie_group_pos", False),
            )
            if name == "pretrained_vit_b16_224_adapter":
                model = flat_moe_vit.flat_moe_vit_base_patch16_224(
                    num_classes=0, global_pool=False, drop_path_rate=0.0,
                    tuning_config=tuning_config)
                model.out_dim = 768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = flat_moe_vit.flat_moe_vit_base_patch16_224_in21k(
                    num_classes=0, global_pool=False, drop_path_rate=0.0,
                    tuning_config=tuning_config)
                model.out_dim = 768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        elif args["model_name"] == "sparse_geo_moe":
            from backbones import sparse_geo_vit
            from easydict import EasyDict
            tuning_config = EasyDict(
                # Sparse Group-MoE
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                ffn_adapter_type=args["ffn_adapter_type"],
                d_model=768,
                attn_bn=ffn_num,
                # VPT
                vpt_on=False,
                vpt_num=0,
                # Sparse Group-MoE specific
                exp_threshold=args["exp_threshold"],
                adapt_start_layer=args["adapt_start_layer"],
                adapt_end_layer=args["adapt_end_layer"],
                rd_dim=args["rd_dim"],
                buffer_size=args["buffer_size"],
                expansion_patience=args.get("expansion_patience", 3),
                # Sparse group routing
                num_geo_groups=args.get("num_geo_groups", 4),
                sparse_top_k=args.get("sparse_top_k", 2),
                router_beta=args.get("router_beta", 0.1),
                router_tau=args.get("router_tau", 1.0),
                # MambaFlow
                mamba_d_state=args.get("mamba_d_state", 16),
                mamba_d_conv=args.get("mamba_d_conv", 4),
                mamba_expand=args.get("mamba_expand", 2),
                # Group-Structured Positional Routing
                use_group_pos=args.get("use_group_pos", False),
                num_groups=args.get("num_groups", 4),
                group_pos_scale=args.get("group_pos_scale", 0.1),
                use_lie_group_pos=args.get("use_lie_group_pos", False),
                # Ablation: set false to use SimpleAdapter in all 12 layers
                use_group_moe=args.get("use_group_moe", True),
            )
            if name == "pretrained_vit_b16_224_adapter":
                model = sparse_geo_vit.sparse_geo_vit_base_patch16_224(
                    num_classes=0, global_pool=False, drop_path_rate=0.0,
                    tuning_config=tuning_config)
                model.out_dim = 768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = sparse_geo_vit.sparse_geo_vit_base_patch16_224_in21k(
                    num_classes=0, global_pool=False, drop_path_rate=0.0,
                    tuning_config=tuning_config)
                model.out_dim = 768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        elif args["model_name"] == "sema":
            from backbones import sema_vit
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                ffn_adapter_type=args["ffn_adapter_type"],
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                exp_threshold=args["exp_threshold"],
                adapt_start_layer=args["adapt_start_layer"],
                adapt_end_layer=args["adapt_end_layer"],
                rd_dim=args["rd_dim"],
                buffer_size=args["buffer_size"],
                # Bundle-SEMA (Group-Structured Positional Routing)
                use_group_pos=args.get("use_group_pos", False),
                num_groups=args.get("num_groups", 4),
                group_pos_scale=args.get("group_pos_scale", 0.1),
                use_lie_group_pos=args.get("use_lie_group_pos", False),
                use_bundle_router=args.get("use_bundle_router", True),
                lambda_geo=args.get("lambda_geo", 0.1),
                geo_rd_dim=args.get("geo_rd_dim", 4),
            )
            if name == "pretrained_vit_b16_224_adapter":
                model = sema_vit.sema_vit_base_patch16_224(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = sema_vit.sema_vit_base_patch16_224_in21k(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        elif args["model_name"] == "aper_adapter":
            from backbones import vit_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
            )
            if name == "pretrained_vit_b16_224_adapter":
                model = vit_adapter.vit_base_patch16_224_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = vit_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # L2P
    elif '_l2p' in name:
        if args["model_name"] == "l2p":
            from backbones import vit_l2p
            model = timm.create_model(
                args["backbone_type"],
                pretrained=args["pretrained"],
                num_classes=args["nb_classes"],
                drop_rate=args["drop"],
                drop_path_rate=args["drop_path"],
                drop_block_rate=None,
                prompt_length=args["length"],
                embedding_key=args["embedding_key"],
                prompt_init=args["prompt_key_init"],
                prompt_pool=args["prompt_pool"],
                prompt_key=args["prompt_key"],
                pool_size=args["size"],
                top_k=args["top_k"],
                batchwise_prompt=args["batchwise_prompt"],
                prompt_key_init=args["prompt_key_init"],
                head_type=args["head_type"],
                use_prompt_mask=args["use_prompt_mask"],
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # dualprompt
    elif '_dualprompt' in name:
        if args["model_name"] == "dualprompt":
            from backbones import vit_dualprompt
            model = timm.create_model(
                args["backbone_type"],
                pretrained=args["pretrained"],
                num_classes=args["nb_classes"],
                drop_rate=args["drop"],
                drop_path_rate=args["drop_path"],
                drop_block_rate=None,
                prompt_length=args["length"],
                embedding_key=args["embedding_key"],
                prompt_init=args["prompt_key_init"],
                prompt_pool=args["prompt_pool"],
                prompt_key=args["prompt_key"],
                pool_size=args["size"],
                top_k=args["top_k"],
                batchwise_prompt=args["batchwise_prompt"],
                prompt_key_init=args["prompt_key_init"],
                head_type=args["head_type"],
                use_prompt_mask=args["use_prompt_mask"],
                use_g_prompt=args["use_g_prompt"],
                g_prompt_length=args["g_prompt_length"],
                g_prompt_layer_idx=args["g_prompt_layer_idx"],
                use_prefix_tune_for_g_prompt=args["use_prefix_tune_for_g_prompt"],
                use_e_prompt=args["use_e_prompt"],
                e_prompt_layer_idx=args["e_prompt_layer_idx"],
                use_prefix_tune_for_e_prompt=args["use_prefix_tune_for_e_prompt"],
                same_key_value=args["same_key_value"],
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # Coda_Prompt
    elif '_coda_prompt' in name:
        if args["model_name"] == "coda_prompt":
            from backbones import vit_coda_promtpt
            model = timm.create_model(args["backbone_type"], pretrained=args["pretrained"])
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    """基础网络包装器 —— 所有持续学习网络的基类。

    封装 backbone + fc 分类头的最小结构。
    子类（SEMAVitNet, IncrementalNet, PromptVitNet 等）在此基础上扩展。

    核心职责：
      - 通过 get_backbone() 加载主干网络
      - 管理分类器头 (fc)
      - 提供特征提取 (extract_vector) 和分类 (forward) 接口
    """

    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class IncrementalNet(BaseNet):
    """标准增量学习网络 —— 使用 SimpleLinear 分类头。

    支持动态扩展分类器以容纳新类别。
    可选 GradCAM 可视化（仅 CNN 模式）。
    """

    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x["features"])
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.backbone.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.backbone.last_conv.register_forward_hook(
            forward_hook
        )


class CosineIncrementalNet(BaseNet):
    """余弦增量网络 —— 使用余弦相似度分类头。

    分类器使用 CosineLinear / SplitCosineLinear，
    适用于基于特征归一化的增量学习。
    """
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num  ==  1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc

class DERNet(nn.Module):
    """DER (Dynamic Expansion and Representation) 网络。

    每个任务动态添加一个新的 backbone，所有 backbone 的输出拼接后分类。
    辅助分类器 (aux_fc) 用于新任务的独立分类。
    """
    def __init__(self, args, pretrained):
        super(DERNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.backbones = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.backbones)

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)

        out = self.fc(features)  # {logits: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim :])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        return out

    def update_fc(self, nb_classes):
        if len(self.backbones) == 0:
            self.backbones.append(get_backbone(self.args, self.pretrained))
        else:
            self.backbones.append(get_backbone(self.args, self.pretrained))
            self.backbones[-1].load_state_dict(self.backbones[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.backbones[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_backbone(self):
        for param in self.backbones.parameters():
            param.requires_grad = False
        self.backbones.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.backbones) == 1
        self.backbones[0].load_state_dict(model_infos['backbone'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc

class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        if self.RP_dim is not None:
            feature_dim = self.RP_dim
        else:
            feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})
        return out

class SEMAVitNet(BaseNet):
    """SEMA ViT 专用网络包装器。

    与标准 BaseNet 的关键区别：
      - backbone.forward() 返回字典 {"features", "rd_loss", "added_record"}
        而非单一特征张量
      - forward() 从字典中提取 features 并用 fc 生成 logits
      - rd_loss 和 added_record 保留在输出中供训练循环使用
    """

    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.fc = None
        self.args = args

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        out = self.backbone(x)
        x = out["features"]
        out.update({"logits": self.fc(x)})
        return out

# l2p and dualprompt
class PromptVitNet(nn.Module):
    """Prompt ViT 网络包装器 —— 用于 L2P 和 DualPrompt 方法。

    可选保留原始 backbone 用于提取 cls_features（作为 prompt 选择的 query）。
    forward() 传递 task_id 和 cls_features 给 backbone 的 prompt 选择机制。
    """
    def __init__(self, args, pretrained):
        super(PromptVitNet, self).__init__()
        self.backbone = get_backbone(args, pretrained)
        if args["get_original_backbone"]:
            self.original_backbone = self.get_original_backbone(args)
        else:
            self.original_backbone = None

    def get_original_backbone(self, args):
        return timm.create_model(
            args["backbone_type"],
            pretrained=args["pretrained"],
            num_classes=args["nb_classes"],
            drop_rate=args["drop"],
            drop_path_rate=args["drop_path"],
            drop_block_rate=None,
        ).eval()

    def forward(self, x, task_id=-1, train=False):
        with torch.no_grad():
            if self.original_backbone is not None:
                cls_features = self.original_backbone(x)['pre_logits']
            else:
                cls_features = None

        x = self.backbone(x, task_id=task_id, cls_features=cls_features, train=train)
        return x

# coda_prompt
class CodaPromptVitNet(nn.Module):
    """CODA-Prompt ViT 网络包装器 —— 基于分解注意力的 Prompt 方法。

    结合 CodaPrompt（可学习的 prompt 组件 + 注意力加权组合）。
    forward() 返回分类 logits，训练时额外返回 prompt_loss。
    """
    def __init__(self, args, pretrained):
        super(CodaPromptVitNet, self).__init__()
        self.args = args
        self.backbone = get_backbone(args, pretrained)
        self.fc = nn.Linear(768, args["nb_classes"])
        self.prompt = CodaPrompt(768, args["nb_tasks"], args["prompt_param"])

    def forward(self, x, pen=False, train=False):
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.backbone(x)
                q = q[:,0,:]
            out, prompt_loss = self.backbone(x, prompt=self.prompt, q=q, train=train)
            out = out[:,0,:]
        else:
            out, _ = self.backbone(x)
            out = out[:,0,:]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.fc(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out


class MultiBranchCosineIncrementalNet(BaseNet):
    """多分支余弦增量网络 (APER 系列方法使用)。

    维护两个 backbones 分支（原始预训练 + 微调后的），
    拼接两者的特征进行分类。
    """
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

        print('Clear the backbone in MultiBranchCosineIncrementalNet, since we are using self.backbones with dual branches')
        self.backbone=torch.nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbones = nn.ModuleList()
        self.args=args

        if 'resnet' in args['backbone_type']:
            self.model_type='cnn'
        else:
            self.model_type='vit'

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self._feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]

        features = torch.cat(features, 1)
        out = self.fc(features)
        out.update({"features": features})
        return out


    def construct_dual_branch_network(self, tuned_model):
        if 'ssf' in self.args['backbone_type']:
            newargs=copy.deepcopy(self.args)
            newargs['backbone_type']=newargs['backbone_type'].replace('_ssf','')
            print(newargs['backbone_type'])
            self.backbones.append(get_backbone(newargs))
        elif 'vpt' in self.args['backbone_type']:
            newargs=copy.deepcopy(self.args)
            newargs['backbone_type']=newargs['backbone_type'].replace('_vpt','')
            print(newargs['backbone_type'])
            self.backbones.append(get_backbone(newargs))
        elif 'adapter' in self.args['backbone_type']:
            newargs=copy.deepcopy(self.args)
            newargs['backbone_type']=newargs['backbone_type'].replace('_adapter','')
            print(newargs['backbone_type'])
            self.backbones.append(get_backbone(newargs))
        else:
            self.backbones.append(get_backbone(self.args))

        self.backbones.append(tuned_model.backbone)

        self._feature_dim = self.backbones[0].out_dim * len(self.backbones)
        self.fc=self.generate_fc(self._feature_dim,self.args['init_cls'])


class FOSTERNet(nn.Module):
    """FOSTER (Feature Boosting and Compression) 网络。

    每个任务添加新 backbone 并拼接特征。
    支持旧分类器 (oldfc) 的知识蒸馏和新特征的特征增强 (fe_fc)。
    """
    def __init__(self, args, pretrained):
        super(FOSTERNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.backbones = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.args = args

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.backbones)

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim :])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        self.backbones.append(get_backbone(self.args, self.pretrained))
        if self.out_dim is None:
            self.out_dim = self.backbones[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.backbones[-1].load_state_dict(self.backbones[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_backbone(self):
        for param in self.backbones.parameters():
            param.requires_grad = False
        self.backbones.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format(
                args["dataset"],
                args["seed"],
                args["backbone_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.backbones) == 1
        self.backbones[0].load_state_dict(model_infos['backbone'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc

class AdaptiveNet(nn.Module):
    """自适应网络 (MEMO 方法使用)。

    包含一个任务无关的特征提取器 (TaskAgnosticExtractor)
    和多个任务特定的自适应模块 (AdaptiveExtractors)。
    """
    def __init__(self, args, pretrained):
        super(AdaptiveNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.TaskAgnosticExtractor , _ = get_backbone(args, pretrained)
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractors = nn.ModuleList()
        self.pretrained=pretrained
        self.out_dim=None
        self.fc = None
        self.aux_fc=None
        self.task_sizes = []
        self.args=args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim*len(self.AdaptiveExtractors)

    def extract_vector(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        out=self.fc(features)

        aux_logits=self.aux_fc(features[:,-self.out_dim:])["logits"]

        out.update({"aux_logits":aux_logits,"features":features})
        out.update({"base_features":base_feature_map})
        return out

    def update_fc(self,nb_classes):
        _ , _new_extractor = get_backbone(self.args, self.pretrained)
        if len(self.AdaptiveExtractors)==0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())

        if self.out_dim is None:
            self.out_dim=self.AdaptiveExtractors[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc=self.generate_fc(self.out_dim,new_task_size+1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma

    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format(
                args["dataset"],
                args["seed"],
                args["backbone_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        checkpoint_name = checkpoint_name.replace("memo_", "")
        model_infos = torch.load(checkpoint_name)
        model_dict = model_infos['backbone']
        assert len(self.AdaptiveExtractors) == 1

        base_state_dict = self.TaskAgnosticExtractor.state_dict()
        adap_state_dict = self.AdaptiveExtractors[0].state_dict()

        pretrained_base_dict = {
            k:v
            for k, v in model_dict.items()
            if k in base_state_dict
        }

        pretrained_adap_dict = {
            k:v
            for k, v in model_dict.items()
            if k in adap_state_dict
        }

        base_state_dict.update(pretrained_base_dict)
        adap_state_dict.update(pretrained_adap_dict)

        self.TaskAgnosticExtractor.load_state_dict(base_state_dict)
        self.AdaptiveExtractors[0].load_state_dict(adap_state_dict)
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc
