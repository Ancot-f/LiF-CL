"""
模型工厂 — 根据 model_name 创建对应的 Learner
=============================================

支持所有持续学习方法：
  - sema: SEMA (Self-Expansion with Mixture of Adapters) — CVPR 2025
  - finetune: 简单微调基线
  - simplecil: 简单类增量学习基线
  - l2p: Learning to Prompt
  - dualprompt: DualPrompt (G-Prompt + E-Prompt)
  - coda_prompt: CODA-Prompt (分解注意力 Prompt)
  - memo: MEMO (内存高效模型)
  - icarl: iCaRL (增量分类器与表征学习)
  - der: DER (动态扩展与表征)
  - coil: CoIL (持续不变学习)
  - foster: FOSTER (特征增强与压缩)
  - aper_*: APER 系列的微调/SSF/VPT/Adapter 变体

用法:
    model = factory.get_model(args["model_name"], args)
"""


def get_model(model_name, args):
    """根据模型名称返回对应的 Learner 实例。

    Args:
        model_name: 模型名称（不区分大小写）
        args: 全局参数字典

    Returns:
        Learner: 对应方法的 Learner 实例
    """
    name = model_name.lower()
    if name == "simplecil":
        from models.simplecil import Learner
    elif name == "aper_finetune":
        from models.aper_finetune import Learner
    elif name == "aper_ssf":
        from models.aper_ssf import Learner
    elif name == "aper_vpt":
        from models.aper_vpt import Learner
    elif name == "aper_adapter":
        from models.aper_adapter import Learner
    elif name == "l2p":
        from models.l2p import Learner
    elif name == "dualprompt":
        from models.dualprompt import Learner
    elif name == "coda_prompt":
        from models.coda_prompt import Learner
    elif name == "finetune":
        from models.finetune import Learner
    elif name == "icarl":
        from models.icarl import Learner
    elif name == "der":
        from models.der import Learner
    elif name == "coil":
        from models.coil import Learner
    elif name == "foster":
        from models.foster import Learner
    elif name == "memo":
        from models.memo import Learner
    elif name == "sema":
        from models.sema import Learner
    else:
        assert 0, f"Unknown model_name: {model_name}"
    return Learner(args)
