"""
Group Basis MoE Vision Transformer Backbone
============================================

ViT-B/16 主干, 深层使用 GroupBasisModules。
架构和权重加载逻辑与 flat_moe_vit.py 一致, 仅适配器模块不同。
"""

import os, sys, timm
from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from timm.models.layers import DropPath
from backbones.group_basis_moe import GroupBasisModules
from backbones.sema_geometry import GroupRoutedPositionalEncoding

try:
    import safetensors.torch
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


class Attention(nn.Module):
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

    def _shape(self, t, seq_len, bsz):
        return t.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1,2).contiguous()

    def forward(self, x):
        B,N,C=x.shape
        q=self.q_proj(x); k=self._shape(self.k_proj(x),-1,B).view(B*self.num_heads,-1,self.head_dim)
        v=self._shape(self.v_proj(x),-1,B).view(B*self.num_heads,-1,self.head_dim)
        q=self._shape(q,N,B).view(B*self.num_heads,-1,self.head_dim)
        aw=torch.bmm(q,k.transpose(1,2))*self.scale; aw=F.softmax(aw,dim=-1)
        ao=torch.bmm(self.attn_drop(aw),v)
        ao=ao.view(B,self.num_heads,N,self.head_dim).transpose(1,2).reshape(B,N,C)
        return self.proj_drop(self.proj(ao))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, config=None, layer_id=None, writer=None):
        super().__init__()
        self.config=config; self.layer_id=layer_id
        self.norm1=norm_layer(dim)
        self.attn=Attention(dim,num_heads,qkv_bias,attn_drop,drop)
        self.drop_path=DropPath(drop_path) if drop_path>0 else nn.Identity()
        mhd=int(dim*mlp_ratio)
        self.norm2=norm_layer(dim); self.fc1=nn.Linear(dim,mhd)
        self.fc2=nn.Linear(mhd,dim); self.act=act_layer(); self.mlp_drop=nn.Dropout(drop)
        self.use_gb=(config.ffn_adapt and layer_id>=getattr(config,'adapt_start_layer',9)
                     and layer_id<=getattr(config,'adapt_end_layer',11))
        self.adapter_module=GroupBasisModules(config,layer_id=layer_id,writer=writer) if self.use_gb else None

    def forward(self,x,group_info=None):
        x=x+self.drop_path(self.attn(self.norm1(x)))
        if self.use_gb:
            xmlp=self.mlp_drop(self.act(self.fc1(self.norm2(x))))
            xmlp=self.drop_path(self.mlp_drop(self.fc2(xmlp)))
            u=x; out=self.adapter_module(u,group_info=group_info)
            x=u+xmlp+out["func_out"]; out.update({"blk_out":x}); return out
        else:
            res=x; x=self.mlp_drop(self.act(self.fc1(self.norm2(x))))
            x=self.drop_path(self.mlp_drop(self.fc2(x))); x=res+x
            return {"blk_out":x,"func_out":torch.zeros_like(x),"rd_loss":torch.tensor(0.,device=x.device),
                    "z_scores":None,"group_weights":None,"basis_weights":None,"added":False}


class VisionTransformer(nn.Module):
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, representation_size=None,
                 distilled=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None,
                 optim_config=None, writer=None):
        super().__init__()
        print("I'm using ViT with Group Basis MoE adapters.")
        self.tuning_config=tuning_config; self.num_classes=num_classes
        self.num_features=self.embed_dim=embed_dim; self.num_tokens=2 if distilled else 1
        norm_layer=norm_layer or partial(nn.LayerNorm,eps=1e-6); act_layer=act_layer or nn.GELU
        self.patch_embed=embed_layer(img_size=img_size,patch_size=patch_size,in_chans=in_chans,embed_dim=embed_dim)
        npatch=self.patch_embed.num_patches
        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))
        self.dist_token=nn.Parameter(torch.zeros(1,1,embed_dim)) if distilled else None
        self.pos_embed=nn.Parameter(torch.zeros(1,npatch+self.num_tokens,embed_dim))
        self.pos_drop=nn.Dropout(p=drop_rate)
        dpr=[x.item() for x in torch.linspace(0,drop_path_rate,depth)]
        self.blocks=nn.Sequential(*[Block(embed_dim,num_heads,mlp_ratio,qkv_bias,drop_rate,
            attn_drop_rate,dpr[i],norm_layer=norm_layer,act_layer=act_layer,
            config=tuning_config,layer_id=i,writer=writer) for i in range(depth)])
        self.norm=norm_layer(embed_dim); self.pre_logits=nn.Identity()
        self.head=nn.Linear(self.num_features,num_classes) if num_classes>0 else nn.Identity()
        self.head_dist=None; self.global_pool=global_pool
        if self.global_pool: self.fc_norm=norm_layer(embed_dim); del self.norm
        if tuning_config.vpt_on:
            self.embeddings=nn.ParameterList([nn.Parameter(torch.empty(1,tuning_config.vpt_num,embed_dim)) for _ in range(depth)])
            for e in self.embeddings: torch.nn.init.xavier_uniform_(e.data)
        self.optim_config=optim_config
        self.use_group_pos=getattr(tuning_config,'use_group_pos',False) if tuning_config else False
        self.group_pos_encoder=GroupRoutedPositionalEncoding(npatch+self.num_tokens,embed_dim,
            getattr(tuning_config,'num_groups',4),getattr(tuning_config,'group_pos_scale',0.1),
            getattr(tuning_config,'use_lie_group_pos',False)) if self.use_group_pos else None

    def forward_features(self, x):
        B=x.shape[0]; x=self.patch_embed(x); x=torch.cat((self.cls_token.expand(B,-1,-1),x),dim=1)
        gi=None
        if self.use_group_pos and self.group_pos_encoder:
            gr=self.group_pos_encoder(x,self.pos_embed); x=gr["x"]; gi={"gp":gr["group_pos"],"gw":gr["group_weights"]}
        else: x=x+self.pos_embed
        x=self.pos_drop(x); trd=torch.tensor(0.,device=x.device); ar=[]
        for idx,blk in enumerate(self.blocks):
            if self.tuning_config.vpt_on: x=torch.cat([self.embeddings[idx].expand(B,-1,-1),x],dim=1)
            br=blk(x,group_info=gi); x=br["blk_out"]
            rd=br.get("rd_loss",torch.tensor(0.,device=x.device))
            trd=trd+(rd if torch.is_tensor(rd) else torch.tensor(0.,device=x.device)); ar.append(br.get("added",False))
            if self.tuning_config.vpt_on: x=x[:,self.tuning_config.vpt_num:,:]
            if br.get("added",False): break
        if self.global_pool: x=x[:,1:,:].mean(1); out=self.fc_norm(x)
        else: x=self.norm(x); out=x[:,0]
        return {"features":out,"rd_loss":trd,"added_record":ar}

    def forward(self,x):
        out=self.forward_features(x); out.update({"logits":self.head(out["features"])}); return out


import torch.nn.functional as F

def _load_pretrained(model, state_dict_source, is_safetensors=False):
    if is_safetensors: sd=safetensors.torch.load_file(state_dict_source)
    else: sd=state_dict_source
    for k in list(sd.keys()):
        if 'qkv.weight' in k:
            w=sd.pop(k); sd[k.replace('qkv.weight','q_proj.weight')]=w[:768]
            sd[k.replace('qkv.weight','k_proj.weight')]=w[768:1536]; sd[k.replace('qkv.weight','v_proj.weight')]=w[1536:]
        elif 'qkv.bias' in k:
            b=sd.pop(k); sd[k.replace('qkv.bias','q_proj.bias')]=b[:768]
            sd[k.replace('qkv.bias','k_proj.bias')]=b[768:1536]; sd[k.replace('qkv.bias','v_proj.bias')]=b[1536:]
    for k in list(sd.keys()):
        if 'mlp.fc' in k: sd[k.replace('mlp.','')]=sd.pop(k)
    msg=model.load_state_dict(sd,strict=False); print(msg)
    for n,p in model.named_parameters(): p.requires_grad=n in msg.missing_keys
    model.out_dim=768; return model


def group_basis_vit_base_patch16_224(pretrained=True, tuning_config=None, **kwargs):
    model=VisionTransformer(patch_size=16,embed_dim=768,depth=12,num_heads=12,mlp_ratio=4,
        qkv_bias=True,norm_layer=partial(nn.LayerNorm,eps=1e-6),tuning_config=tuning_config,**kwargs)
    if pretrained:
        if _HAS_SAFETENSORS:
            sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            try:
                from lif_cl.paths import get_premodel_path; pp=get_premodel_path("model.safetensors")
            except: pp="/sdb/syc/My_code/LiF-CL/pre-model/model.safetensors"
            if os.path.exists(pp): model=_load_pretrained(model,pp,True)
            else: print("Falling back to timm"); model=_load_pretrained(model,timm.create_model("vit_base_patch16_224",pretrained=True,num_classes=0).state_dict())
        else: model=_load_pretrained(model,timm.create_model("vit_base_patch16_224",pretrained=True,num_classes=0).state_dict())
    return model
