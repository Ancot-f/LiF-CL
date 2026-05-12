"""
Lie-Group FiberCL — 入口脚本
============================

使用 lif_cl 共享库进行:
- YAML 配置加载 (ConfigManager)
- 自动 GPU 选择 (set_seed, set_device)
- wandb 实验追踪 (WandbLogger)
- 断点续跑 (CheckpointManager)
- 损失追踪 (LossTracker)
- 路径解析 (lif_cl.paths)

SEMA 训练流程:
  每个任务 → 训练 → 评估 → 保存检查点 → 上报 wandb
"""

import sys
import os
import logging
import copy
import argparse
import textwrap
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lif_cl.config import ConfigManager
from lif_cl.seed import set_seed, set_device
from lif_cl.wandb_logger import WandbLogger
from lif_cl.checkpoint import CheckpointManager

from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from utils import factory
from backbones.sema_block import SEMAModules

# ── 表格绘制常量 ──
_BOX_H  = "═"
_BOX_V  = "║"
_BOX_TL = "╔"; _BOX_TR = "╗"; _BOX_BL = "╚"; _BOX_BR = "╝"
_BOX_ML = "╠"; _BOX_MR = "╣"; _BOX_MM = "╬"
_SEP_L  = "╟"; _SEP_R  = "╢"


def main():
    """主入口: 加载配置 → 初始化 → 训练循环。"""
    parser = argparse.ArgumentParser(
        description="Lie-Group FiberCL — SEMA 持续学习训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            使用示例:
              # 使用默认配置 (configs/sema.yaml)
              python main.py

              # 指定配置文件
              python main.py --config configs/sema.yaml

              # 覆盖参数
              python main.py --config configs/sema.yaml lr=0.01 seed=[99] device=[0]

              # 断点续跑
              python main.py --config configs/sema.yaml resume=true

              # 查看所有可用数据集
              python main.py --list-datasets
            """),
    )
    parser.add_argument("--config", type=str, default="configs/sema.yaml",
                        help="YAML 配置文件路径 (默认: configs/sema.yaml)")
    parser.add_argument("--list-datasets", action="store_true",
                        help="列出所有可用数据集")
    parser.add_argument("overrides", nargs="*", default=[],
                        help='覆盖配置值，如 lr=0.01 seed=[99] device=[0]')
    args = parser.parse_args()

    # 列出数据集
    if args.list_datasets:
        _print_available_datasets()
        return

    # 解析 CLI 覆盖参数
    overrides = {}
    for override in args.overrides:
        if "=" in override:
            key, val = override.split("=", 1)
            overrides[key] = val

    cfg = ConfigManager(args.config, cli_overrides=overrides if overrides else None)
    params = cfg.to_dict()

    # 多 seed 训练循环
    seed_list = copy.deepcopy(params["seed"])
    for seed in seed_list:
        params["seed"] = seed
        _train_single(params)


def _train_single(args):
    """单 seed 训练。

    完整的训练管线:
    setup → model init → wandb → checkpoint → training loop → summary
    """
    # ---- 路径和日志 ----
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"], args["dataset"], init_cls, args["increment"],
        args["prefix"], args["seed"], args["backbone_type"],
    )
    os.makedirs(os.path.dirname(logfilename), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # ---- 设备 + 种子 ----
    set_seed(args["seed"])
    args["device"] = set_device(args["device"])

    # ---- 数据 ----
    data_manager = DataManager(
        args["dataset"], args["shuffle"], args["seed"],
        args["init_cls"], args["increment"], args,
    )
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks

    # ---- 模型 ----
    model = factory.get_model(args["model_name"], args)

    # ── 打印模型结构与超参数 ──
    _print_model_structure(model)
    _print_hyperparams_table(args)

    # ---- 断点续跑 ----
    ckpt_dir = args.get("checkpoint_dir") or "checkpoints/{}/{}/{}_{}".format(
        args["model_name"], args["dataset"], args["prefix"], args["seed"]
    )
    ckpt = CheckpointManager(ckpt_dir)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    wandb_run_id = None
    start_task = 0

    if args.get("resume") and os.path.exists(ckpt_dir):
        state = ckpt.load_latest()
        if state is not None:
            logging.info("Resuming from checkpoint: task %d", state["task_id"])
            # 从 checkpoint 推断适配器结构, 重建后再加载权重
            _rebuild_adapters_from_state(model._network, state["model"])
            model._network.load_state_dict(state["model"])
            if state.get("data_memory") is not None:
                model._data_memory, model._targets_memory = state["data_memory"]
            model._known_classes = state["known_classes"]
            model._total_classes = state["total_classes"]
            model._cur_task = state["task_id"]
            if state.get("metrics"):
                m = state["metrics"]
                cnn_curve = m.get("cnn_curve", cnn_curve)
                nme_curve = m.get("nme_curve", nme_curve)
                cnn_matrix = m.get("cnn_matrix", cnn_matrix)
                nme_matrix = m.get("nme_matrix", nme_matrix)
            wandb_run_id = state.get("wandb_run_id")
            start_task = state["task_id"] + 1
            logging.info("Will continue from task %d/%d", start_task, data_manager.nb_tasks)

    # ---- wandb ----
    wandb_cfg = args.get("wandb", {})
    wandb_mode = wandb_cfg.get("mode", "online")
    import os as _os
    _os.environ["WANDB_MODE"] = wandb_mode
    if wandb_mode == "disabled":
        _os.environ["WANDB_SILENT"] = "true"
    wandb_logger = WandbLogger(
        project=wandb_cfg.get("project", "LiF-CL"),
        group=wandb_cfg.get("group", "SEMA"),
        name=logfilename.replace("/", "_"),
        config=args,
        tags=wandb_cfg.get("tags", []),
        resume_id=wandb_run_id,
    )
    # 类增量曲线: X轴 = 累计类别数, Y轴 = 精度/遗忘率
    wandb_logger.define_metric("eval/total_classes")
    wandb_logger.define_metric("eval/*", step_metric="eval/total_classes")
    wandb_logger.define_metric("loss/*", step_metric=None)  # loss 用默认 step
    model.set_wandb_logger(wandb_logger)

    # ---- 训练循环 ----
    for task in range(start_task, data_manager.nb_tasks):
        total_params = count_parameters(model._network)
        trainable_params = count_parameters(model._network, True)
        logging.info("All params: {}".format(total_params))
        logging.info("Trainable params: {}".format(trainable_params))

        # 增量训练
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        # 累积评估指标
        cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
        cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
        cnn_matrix.append(cnn_values)
        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])

        if nme_accy is not None:
            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)
            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

        # 计算指标
        avg_acc = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
        acc_old = cnn_accy["grouped"].get("old", 0)
        acc_new = cnn_accy["grouped"].get("new", 0)
        forgetting = _compute_forgetting(cnn_matrix)

        # 日志
        logging.info("CNN: {}".format(cnn_accy["grouped"]))
        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("Forgetting: {:.2f}".format(forgetting))
        logging.info("Average Accuracy: {:.2f}".format(avg_acc))

        # 累计已见类别数 (类增量曲线的 X 轴)
        cumulative_classes = model._total_classes

        # wandb 上报（论文风格）
        wandb_metrics = {
            "eval/total_classes": cumulative_classes,  # X 轴: 累计类别数
            "eval/acc_avg": avg_acc,                   # 平均精度
            "eval/acc_current": cnn_accy["top1"],       # 当前精度
            "eval/acc_old": acc_old,                   # 旧类精度
            "eval/acc_new": acc_new,                   # 新类精度
            "eval/cnn_top5": cnn_accy["top5"],         # Top-5 精度
            "eval/forgetting": forgetting,              # 遗忘率
            "expansion/total_params": total_params,
            "expansion/trainable_params": trainable_params,
            "expansion/param_ratio": trainable_params / total_params if total_params > 0 else 0,
        }
        if nme_accy is not None:
            wandb_metrics["eval/nme_top1"] = nme_accy["top1"]
            wandb_metrics["eval/nme_top5"] = nme_accy["top5"]

        wandb_logger.log_metrics(wandb_metrics)  # step 自动递增, 避免与 LossTracker 冲突

        # 保存检查点
        ckpt.save_task_state(
            task_id=task,
            model=model._network,
            data_memory=(model._data_memory, model._targets_memory)
                if len(model._data_memory) > 0 else None,
            known_classes=model._known_classes,
            total_classes=model._total_classes,
            metrics={"cnn_curve": cnn_curve, "nme_curve": nme_curve,
                     "cnn_matrix": cnn_matrix, "nme_matrix": nme_matrix},
            wandb_run=wandb_logger.run,
            config=args,
        )
        logging.info("Checkpoint saved: %s/task_%02d", ckpt_dir, task)

    # ---- 训练结束 ----
    wandb_logger.log_task_matrix(cnn_matrix)

    final_avg_acc = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
    final_forgetting = _compute_forgetting(cnn_matrix)
    wandb_logger.log_summary({
        "eval/avg_acc": final_avg_acc,
        "eval/final_acc": cnn_curve["top1"][-1],
        "eval/forgetting": final_forgetting,
    })
    if nme_curve["top1"]:
        nme_avg = sum(nme_curve["top1"]) / len(nme_curve["top1"])
        wandb_logger.log_summary({
            "eval/nme_avg_acc": nme_avg,
            "eval/nme_final_acc": nme_curve["top1"][-1],
        })

    logging.info("Training complete. Average Accuracy (CNN): {:.2f}".format(final_avg_acc))
    wandb_logger.finish()


def _rebuild_adapters_from_state(network, state_dict):
    """从 checkpoint 推断适配器结构并重建。

    checkpoint 只保存权重, 不保存适配器数量。
    需要从 state_dict 的 key 名推断每层有多少个适配器,
    然后调用 add_adapter() + end_of_task_training() 重建结构。

    同时处理分类器 fc 的维度变化。
    """
    import re
    import torch.nn as nn
    from backbones.sema_block import SEMAModules

    # 推断每层适配器数量
    adapter_pattern = [1] * 12
    pattern = re.compile(r'backbone\.blocks\.(\d+)\.adapter_module\.adapters\.(\d+)\.')
    for key in state_dict.keys():
        match = pattern.search(key)
        if match:
            block_id = int(match.group(1))
            adapter_id = int(match.group(2))
            adapter_pattern[block_id] = max(adapter_pattern[block_id], adapter_id + 1)

    # 重建适配器
    for block_id, num in enumerate(adapter_pattern):
        module = network.backbone.blocks[block_id].adapter_module
        while module.num_adapters < num:
            module.add_adapter()              # 添加新适配器(创建 new_router)
            module.freeze_functional()        # fix_router 合并路由列, 冻结功能模块
            module.freeze_rd()                # 冻结表征描述器
            module.reset_newly_added_status() # 清除 newly_added 标志
            module.added_for_task = False     # 允许后续继续添加
        logging.info("Block %d: rebuilt to %d adapters", block_id, num)

    # 重建分类器 fc (如果 checkpoint 中 fc 维度与当前不同)
    if "fc.weight" in state_dict:
        fc_weight = state_dict["fc.weight"]
        if network.fc is None or network.fc.out_features != fc_weight.shape[0]:
            network.fc = nn.Linear(768, fc_weight.shape[0])
            logging.info("Rebuilt fc: 768 -> %d", fc_weight.shape[0])


def _compute_forgetting(cnn_matrix):
    """标准 CL 遗忘率 (Chaudhry et al.)。

    对每个已见过的任务组 j:
        forgetting_j = 该组的最高准确率 - 该组的当前准确率
    总遗忘率 = 所有组的平均遗忘率

    Args:
        cnn_matrix: 分组准确率矩阵 matrix[k][j] = 第 k 个任务后第 j 组的准确率
    Returns:
        float: 平均遗忘率（值越小越好）
    """
    if len(cnn_matrix) < 2:
        return 0.0
    num_groups = len(cnn_matrix[-1])
    total_forgetting = 0.0
    for j in range(num_groups):
        peak = max(row[j] for row in cnn_matrix if j < len(row))
        final = cnn_matrix[-1][j] if j < len(cnn_matrix[-1]) else peak
        total_forgetting += max(0.0, peak - final)
    return total_forgetting / num_groups


# ═══════════════════════════════════════════════════════════════════════════
#  终端格式化输出 (line-list pattern: 数据与渲染分离)
# ═══════════════════════════════════════════════════════════════════════════

_W = 80  # 统一宽度


def _box_line(left, right, fill):
    """边框行。"""
    return left + fill * (_W - 2) + right


def _box_header(title):
    """带标题的头部行。"""
    return _box_line(_BOX_TL, _BOX_TR, _BOX_H) + "\n" + \
           _BOX_V + _visual_ljust(f"  {title}", _W - 2) + _BOX_V


def _box_section(title):
    """节分隔行。"""
    return _BOX_V + _visual_ljust(f"── {title} ", _W - 2, fill="─") + _BOX_V


_KW = 18  # key column width


def _kv(key, value):
    """Two-column key-value row: '  key              value'."""
    return f"  {key:<{_KW}s}  {str(value)}"


def _render_box(title, lines, width=None):
    """将行列表渲染为带边框的盒子。

    Args:
        title: 盒子标题 (可为 None)
        lines: 行列表
        width: 总宽度（默认 _W=80）
    """
    if width is None:
        width = _W
    inner = width - 2
    print()
    if title:
        print(_box_header(title))
    else:
        print(_box_line(_BOX_TL, _BOX_TR, _BOX_H))

    print(_box_line(_BOX_ML, _BOX_MR, _BOX_H))
    for line in lines:
        if isinstance(line, tuple):
            tag = line[0]
            if tag == "sep":
                print(_box_section(line[1]))
            elif tag == "kv":
                print(_BOX_V + _visual_ljust(_kv(line[1], line[2]), inner) + _BOX_V)
        elif line == "":
            print(_BOX_V + " " * inner + _BOX_V)
        else:
            print(_BOX_V + _visual_ljust(line, inner) + _BOX_V)

    print(_box_line(_BOX_BL, _BOX_BR, _BOX_H))
    print()


def _print_available_datasets():
    """列出可用数据集。"""
    datasets = [
        ("cifar10",   "10 类, 5 tasks × 2/task",   "cifar-10-batches-py"),
        ("cifar100",  "100 类, 10 tasks × 10/task", "cifar-100-python"),
        ("imagenetr", "200 类, 20 tasks × 10/task", "imagenet-r/train/"),
        ("imageneta", "200 类, 20 tasks × 10/task", "imagenet-a/train/"),
        ("vtab",      "50 类, 5 tasks × 10/task",   "vtab/train/"),
    ]
    lines = []
    for name, desc, path in datasets:
        lines.append(f"  {name:<12s} {desc:<28s} {path}")
    _render_box("Available Datasets", lines)
    print("  路径自动解析自 LiF-CL/dataset/，可通过 config 中的 data_path 覆盖。")
    print()


def _print_model_structure(model):
    """Print model structure breakdown."""
    net = model._network
    total_p = count_parameters(net)
    trainable_p = count_parameters(net, True)
    backbone_p = count_parameters(net.backbone) if hasattr(net, 'backbone') else 0
    fc_p = count_parameters(net.fc) if hasattr(net, 'fc') and net.fc is not None else 0
    adapter_p = sum(count_parameters(m) for m in net.modules() if isinstance(m, SEMAModules))

    sema_modules = [m for m in net.modules() if isinstance(m, SEMAModules)]
    total_adapters = sum(m.num_adapters for m in sema_modules)
    nb_tasks = model.args.get("nb_tasks", "?")
    nb_classes = model.args.get("nb_classes", "?")
    start_l = model.args.get("adapt_start_layer", "?")
    end_l = model.args.get("adapt_end_layer", "?")

    lines = [
        ("sep", "Components"),
        ("kv", "backbone",       f"ViT-B/16 + SEMA blocks"
                                 f"     FROZEN     {backbone_p:>12,} params"),
        ("kv", "  adapters",     f"{len(sema_modules)} layers, {total_adapters} adapters"
                                 f"                {adapter_p:>12,} params"),
        ("kv", "classifier",     f"Linear  768 -> {nb_classes}"
                                 f"            TRAINABLE   {fc_p:>12,} params"),
        "",
        ("kv", "TOTAL",          f"{total_p:>12,} params"
                                 f"     trainable: {trainable_p:,} ({trainable_p * 100 // total_p}%)"),
        "",
        ("sep", "SEMA Config"),
        ("kv", "adapt-layers",   f"L{start_l} -> L{end_l}  ({end_l - start_l + 1} layers)"),
        ("kv", "exp-threshold",  f"z-score > {model.args.get('exp_threshold', '?')}"),
        ("kv", "rd-dim",         str(model.args.get("rd_dim", "?"))),
        ("kv", "replay",         "ON" if model.args.get("memory_size", 0) > 0 else "OFF"),
        ("kv", "tasks",          str(nb_tasks)),
    ]
    _render_box("Model Structure", lines, width=100)


def _print_hyperparams_table(args):
    """Print hyperparameter table."""
    dev = args.get("device", "?")
    if isinstance(dev, list):
        dev = str(dev[0])

    lines = [
        ("sep", "Experiment"),
        ("kv", "method",        args.get("model_name", "?")),
        ("kv", "device",        dev),
        ("kv", "seed",          str(args.get("seed", "?"))),
        ("kv", "save-root",     "./checkpoints"),
        "",
        ("sep", "Dataset"),
        ("kv", "dataset",       f"{args.get('dataset', '?')}  ({args.get('nb_classes', '?')} classes, "
                                f"{args.get('nb_tasks', '?')} tasks x {args.get('increment', '?')}/task)"),
        ("kv", "img-size",      "224"),
        ("kv", "data-root",     args.get("data_path", "<auto>")),
        "",
        ("sep", "Model — SEMA"),
        ("kv", "backbone",      "vit-b16-224  pretrained=True  dim=768  FROZEN"),
        ("kv", "adapters",      f"init=1/layer  bottleneck={args.get('ffn_num', '?')}  "
                                f"detect=L{args.get('adapt_start_layer', '?')}-L{args.get('adapt_end_layer', '?')}"),
        ("kv", "rd-dim",        str(args.get("rd_dim", "?"))),
        ("kv", "exp-threshold", f"z > {args.get('exp_threshold', '?')}"),
        ("kv", "classifier",    f"linear  768 -> {args.get('nb_classes', '?')}"),
        "",
        ("sep", "Training"),
        ("kv", "optimizer",     args.get("optimizer", "?")),
        ("kv", "lr",            f"func={args.get('init_lr', '?')}  rd={args.get('rd_lr', '?')}"),
        ("kv", "weight-decay",  str(args.get("weight_decay", "?"))),
        ("kv", "batch-size",    str(args.get("batch_size", "?"))),
        ("kv", "epochs",        f"func={args.get('func_epoch', '?')}  rd={args.get('rd_epoch', '?')}"),
        ("kv", "min-lr",        str(args.get("min_lr", "?"))),
        ("kv", "amp",           "True"),
        "",
        ("sep", "Loss"),
        ("kv", "CE  (func)",    "1.0"),
        ("kv", "RD  (rd)",      "1.0"),
    ]
    _render_box("Hyperparameters", lines, width=100)


def _visual_width(s):
    """计算字符串的终端视觉宽度（中文=2列，ASCII=1列）。"""
    w = 0
    for ch in s:
        w += 2 if '一' <= ch <= '鿿' or '　' <= ch <= '〿' or '＀' <= ch <= '￯' else 1
    return w


def _visual_ljust(s, width, fill=" "):
    """视觉宽度左对齐，用 fill 字符填充到 width 列。"""
    return s + fill * max(0, width - _visual_width(s))


def _visual_center(s, width):
    """视觉宽度居中。"""
    vw = _visual_width(s)
    left = max(0, (width - vw) // 2)
    return ' ' * left + s


def _print_banner():
    """打印启动横幅和快速开始指南。"""
    W = 80
    lines = [
        ("title", "Lie-Group FiberCL — 李群纤维丛持续学习"),
        ("gap", ""),
        ("sub", "基于 lif_cl 框架, 集成 SEMA (CVPR 2025) 自扩展适配器方法"),
        ("gap", ""),
        ("head", "快速开始"),
        ("cmd",  "python main.py",                                       "使用默认配置"),
        ("cmd",  "python main.py --config configs/sema.yaml",            "指定配置文件"),
        ("cmd",  "python main.py --config configs/sema.yaml  lr=0.01 device=[0]", "覆盖参数"),
        ("cmd",  "python main.py --config configs/sema.yaml resume=true",  "断点续跑 (自动检测最新 checkpoint)"),
        ("cmd",  "python main.py --list-datasets",                       "查看可用数据集"),
        ("", ""),
        ("head", "配置文件"),
        ("text", "configs/sema.yaml   SEMA (CVPR 2025) 标准配置"),
        ("", ""),
        ("head", "断点续跑"),
        ("text", "resume=true                                    自动从 checkpoints/ 恢复最新任务"),
        ("text", "resume=true checkpoint_dir=checkpoints/sema/...  指定 checkpoint 目录"),
        ("text", "checkpoint 自动保存每个任务后的模型/内存/指标, 中断后可无缝继续"),
        ("", ""),
        ("head", "常用覆盖参数"),
        ("text", "dataset=cifar100    init_cls=10        increment=10"),
        ("text", "device=[0]          seed=[1993]        batch_size=32"),
        ("text", "func_epoch=5        rd_epoch=20        exp_threshold=2"),
        ("text", "lr=0.01"),
    ]

    print()
    print(_box_line(_BOX_TL, _BOX_TR, _BOX_H))
    for tag, a, *rest in lines:
        if tag in ("", "gap"):
            print(_BOX_V + " " * (W - 2) + _BOX_V)
        elif tag in ("title", "sub"):
            print(_BOX_V + _visual_center(a, W - 2) + _BOX_V)
        elif tag == "head":
            print(_BOX_V + "  " + _visual_ljust(a, W - 4) + _BOX_V)
        elif tag == "cmd":
            b = rest[0] if rest else ""
            body = f"  {a:<52s}{'# ' + b}"
            print(_BOX_V + _visual_ljust(body, W - 2) + _BOX_V)
        elif tag == "text":
            print(_BOX_V + "  " + _visual_ljust(a, W - 4) + _BOX_V)
    print(_box_line(_BOX_BL, _BOX_BR, _BOX_H))
    print()


if __name__ == "__main__":
    # 无参数启动时打印帮助
    if len(sys.argv) == 1:
        _print_banner()
    else:
        main()
