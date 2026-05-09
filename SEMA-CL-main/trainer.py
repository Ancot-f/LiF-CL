import sys
import os
import logging
import copy
import torch
import numpy as np
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lif_cl.wandb_logger import WandbLogger
from lif_cl.checkpoint import CheckpointManager
from lif_cl.seed import set_seed, set_device


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"]
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

    set_seed(args["seed"])
    args["device"] = set_device(args["device"])
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )

    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    # Checkpoint
    ckpt_dir = args.get("checkpoint_dir") or "checkpoints/{}/{}/{}_{}".format(
        args["model_name"], args["dataset"], args["prefix"], args["seed"]
    )
    ckpt = CheckpointManager(ckpt_dir)

    # Resume or start fresh
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    wandb_run_id = None
    start_task = 0

    if args.get("resume") and os.path.exists(ckpt_dir):
        state = ckpt.load_latest()
        if state is not None:
            logging.info("Resuming from checkpoint: task %d", state["task_id"])
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

    # wandb
    wandb_cfg = args.get("wandb", {})
    wandb_logger = WandbLogger(
        project=wandb_cfg.get("project", "LiF-CL"),
        group="SEMA",
        name=logfilename.replace("/", "_"),
        config=args,
        tags=wandb_cfg.get("tags", []),
        resume_id=wandb_run_id,
    )
    model.set_wandb_logger(wandb_logger)

    acc_matrix = []  # [task_i][task_j] = accuracy on task j after finishing task i

    for task in range(start_task, data_manager.nb_tasks):
        total_params = count_parameters(model._network)
        trainable_params = count_parameters(model._network, True)
        logging.info("All params: {}".format(total_params))
        logging.info("Trainable params: {}".format(trainable_params))

        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        # ---- Build accuracy matrix row for this task ----
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

        # Compute metrics
        avg_acc = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
        acc_old = cnn_accy["grouped"].get("old", 0)
        acc_new = cnn_accy["grouped"].get("new", 0)

        # Forgetting measure: average drop from peak to current, per old task
        forgetting = _compute_forgetting(cnn_matrix)

        # Logging
        logging.info("CNN: {}".format(cnn_accy["grouped"]))
        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("Forgetting: {:.2f}".format(forgetting))
        print('Average Accuracy (CNN): {:.2f}'.format(avg_acc))
        logging.info("Average Accuracy (CNN): {}".format(avg_acc))

        # ---- Wandb logging (paper-style) ----
        wandb_metrics = {
            "eval/acc_avg": avg_acc,
            "eval/acc_old": acc_old,
            "eval/acc_new": acc_new,
            "eval/cnn_top1": cnn_accy["top1"],
            "eval/cnn_top5": cnn_accy["top5"],
            "eval/forgetting": forgetting,
            "expansion/total_params": total_params,
            "expansion/trainable_params": trainable_params,
            "expansion/param_ratio": trainable_params / total_params if total_params > 0 else 0,
        }
        if nme_accy is not None:
            wandb_metrics["eval/nme_top1"] = nme_accy["top1"]
            wandb_metrics["eval/nme_top5"] = nme_accy["top5"]

        wandb_logger.log_metrics(wandb_metrics, step=task)

        # Save checkpoint
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

    # Final task accuracy matrix (one table, heatmap style)
    wandb_logger.log_task_matrix(cnn_matrix)

    # Summary
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
    wandb_logger.finish()


def _compute_forgetting(cnn_matrix):
    """Standard CL forgetting measure (Chaudhry et al.).

    For each task group j, forgetting_j = peak_acc_j - final_acc_j.
    Returns the average across all groups seen so far.

    Args:
        cnn_matrix: list of lists, matrix[k][j] = accuracy on group j after task k.
    Returns:
        float: average forgetting
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


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))