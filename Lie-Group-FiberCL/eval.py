"""
SEMA 独立评估脚本
================
从检查点加载已训练的 SEMA 模型，在完整测试集上评估。

用法：
    python eval.py --config exps/sema_inr_10task.json --eval true --checkpt_path checkpoints/xxx.pth

核心流程：
    1. 加载配置和数据集
    2. 创建模型
    3. 从检查点读取各层适配器数量（adapter_pattern）
    4. 重建适配器结构并加载权重
    5. 在所有已见类别上评估
"""

import sys
import logging
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import re
import numpy as np

def eval(args):
    """从检查点评估已训练的 SEMA 模型。

    Args:
        args: 参数字典，需包含 checkpt_path（检查点文件路径）
    """
    args["seed"] = args["seed"][0]
    device = copy.deepcopy(args["device"])

    # ---- 日志配置 ----
    logs_name = "logs/{}/{}".format(args["model_name"], args["backbone_type"])
    logfilename = "logs/{}/{}/eval_{}".format(
        args["model_name"],
        args["backbone_type"],
        args["dataset"],
    )

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # ---- 环境设置 ----
    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    # ---- 数据加载 ----
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )

    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks

    # ---- 模型创建 ----
    model = factory.get_model(args["model_name"], args)

    # 设置模型状态为最后一个任务完成时
    model._cur_task = data_manager.nb_tasks - 1
    model._network.fc = nn.Linear(768, data_manager.nb_classes)
    model._total_classes = data_manager.nb_classes

    # 构建测试数据加载器（所有已见类别）
    test_dataset = data_manager.get_dataset(
        np.arange(0, model._total_classes), source="test", mode="test"
    )
    model.test_loader = DataLoader(
        test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=8
    )

    # ---- 从检查点重建适配器结构 ----
    # 检查点只保存适配器权重，不保存适配器数量
    # 需要从 state_dict 的 key 中推断每层有多少个适配器
    adapter_pattern = get_adapter_pattern(args["checkpt_path"])

    # 按 pattern 重建：为每层添加对应数量的适配器并冻结
    for idx, n in enumerate(adapter_pattern):
        if n > 1:  # 第一层有 > 1 个适配器（说明训练中触发了扩展）
            for i in range(n - 1):  # 添加额外的 n-1 个适配器
                model._network.backbone.blocks[idx].adapter_module.add_adapter()
                model._network.backbone.blocks[idx].adapter_module.end_of_task_training()

    # 加载保存的适配器权重（strict=False 因为分类器头可能不匹配）
    model.load_checkpoint(args["checkpt_path"])
    model._network.to(args["device"][0])

    # ---- 评估 ----
    cnn_accy, _ = model.eval_task()
    logging.info("CNN: {}".format(cnn_accy["grouped"]))


def get_adapter_pattern(checkpt_path):
    """从检查点推断每层适配器数量。

    解析 state_dict 的 key 名找出每层最大的 adapter_id，
    从而重建训练时的适配器结构。

    例如 key "backbone.blocks.9.adapter_module.adapters.2.functional.down_proj.weight"
    表示第9层有至少 3 个适配器（索引 0, 1, 2）。

    Args:
        checkpt_path: 检查点文件路径

    Returns:
        list[int]: 长度为 12 的列表，pattern[i] = 第 i 层的适配器数量
    """
    state_dict = torch.load(checkpt_path)
    adapter_pattern = [1] * 12  # 默认每层 1 个适配器

    # 匹配模式：backbone.blocks.{层号}.adapter_module.adapters.{适配器号}.
    pattern = re.compile(r'backbone\.blocks\.(\d+)\.adapter_module\.adapters\.(\d+)\.')

    for key in state_dict.keys():
        match = pattern.search(key)
        if match:
            block_id = int(match.group(1))     # 层号 (0-11)
            adapter_id = int(match.group(2))   # 适配器号 (0-based)
            adapter_pattern[block_id] = max(adapter_pattern[block_id], adapter_id + 1)

    return adapter_pattern


def _set_device(args):
    """设置计算设备（CPU 或 GPU）。

    将 args["device"] 中的设备标识符转换为 torch.device 对象。
    """
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    """固定随机种子，确保评估可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    """打印所有参数（日志记录）。"""
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
