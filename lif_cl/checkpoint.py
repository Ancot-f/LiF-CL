"""Checkpoint manager for resumable continual learning training.

Saves full training state so that if a run crashes, it can be resumed
from the last completed task without losing progress.

Usage:
    # --- Save after each task ---
    ckpt = CheckpointManager("checkpoints/sema/cifar100/run_001")
    ckpt.save_task_state(
        task_id=3,
        model=model._network,
        optimizer=optimizer,
        scheduler=scheduler,
        known_classes=150,
        total_classes=150,
        data_memory=(model._data_memory, model._targets_memory),
        metrics={"cnn_curve": cnn_curve, "cnn_matrix": cnn_matrix},
        wandb_run=wandb_logger.run,
        config=args,
    )

    # --- Resume ---
    ckpt = CheckpointManager("checkpoints/sema/cifar100/run_001")
    state = ckpt.load_latest()
    if state:
        model._network.load_state_dict(state["model"])
        model._data_memory, model._targets_memory = state["data_memory"]
        model._known_classes = state["known_classes"]
        start_task = state["task_id"] + 1  # resume from next task
        cnn_curve = state["metrics"]["cnn_curve"]
"""

import json
import os
import shutil
import warnings
from datetime import datetime

import numpy as np
import torch


class CheckpointManager:
    """Manages training checkpoints for crash recovery.

    Each checkpoint is a directory containing model weights, optimizer state,
    and a metadata file tracking what has been completed.
    """

    def __init__(self, save_dir, keep_last_n=3):
        self._save_dir = save_dir
        self._keep_last_n = keep_last_n
        os.makedirs(save_dir, exist_ok=True)

    @property
    def save_dir(self):
        return self._save_dir

    def save_task_state(
        self,
        task_id,
        model=None,
        optimizer=None,
        scheduler=None,
        known_classes=None,
        total_classes=None,
        data_memory=None,
        metrics=None,
        wandb_run=None,
        config=None,
        extra_state=None,
    ):
        """Save full state after completing a task.

        Args:
            task_id: current task index (0-based)
            model: nn.Module or state_dict
            optimizer: torch optimizer
            scheduler: torch lr scheduler
            known_classes: int
            total_classes: int
            data_memory: tuple of (data, targets) numpy arrays (exemplars)
            metrics: dict of accumulated metrics so far
            wandb_run: wandb.run object (to resume same run)
            config: dict of all hyperparams
            extra_state: dict of any additional state to preserve
        """
        task_dir = os.path.join(self._save_dir, f"task_{task_id:02d}")
        os.makedirs(task_dir, exist_ok=True)

        # Model
        if model is not None:
            state_dict = model.state_dict() if hasattr(model, "state_dict") else model
            torch.save(state_dict, os.path.join(task_dir, "model.pt"))

        # Optimizer
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(task_dir, "optimizer.pt"))

        # Scheduler
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(task_dir, "scheduler.pt"))

        # Exemplar memory
        if data_memory is not None:
            data, targets = data_memory
            np.savez(
                os.path.join(task_dir, "memory.npz"),
                data=data, targets=targets,
            )

        # Extra state
        if extra_state:
            torch.save(extra_state, os.path.join(task_dir, "extra.pt"))

        # Metadata
        metadata = {
            "task_id": task_id,
            "known_classes": known_classes,
            "total_classes": total_classes,
            "timestamp": datetime.now().isoformat(),
            "wandb_run_id": wandb_run.id if (wandb_run and hasattr(wandb_run, "id")) else None,
            "metrics": metrics,
            "config": _sanitize_config(config) if config else None,
        }
        with open(os.path.join(task_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=_json_default)

        # Update 'latest' symlink
        latest_link = os.path.join(self._save_dir, "latest")
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(f"task_{task_id:02d}", latest_link)

        # Cleanup old checkpoints
        self._cleanup_old(task_id)

    def _cleanup_old(self, current_task):
        """Remove old task checkpoints, keeping only the last `keep_last_n`."""
        if self._keep_last_n <= 0:
            return
        keep_from = max(0, current_task - self._keep_last_n + 1)
        for task_dir_name in os.listdir(self._save_dir):
            if task_dir_name.startswith("task_") and task_dir_name != "latest":
                try:
                    t = int(task_dir_name.split("_")[1])
                    if t < keep_from:
                        shutil.rmtree(os.path.join(self._save_dir, task_dir_name))
                except (ValueError, IndexError):
                    pass

    def save_epoch_state(self, task_id, epoch, model, optimizer, scheduler):
        """Save state within a task (for intra-task resume). Saves to a single rotating file."""
        epoch_dir = os.path.join(self._save_dir, f"task_{task_id:02d}_epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        if model is not None:
            sd = model.state_dict() if hasattr(model, "state_dict") else model
            torch.save(sd, os.path.join(epoch_dir, "model.pt"))
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(epoch_dir, "optimizer.pt"))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(epoch_dir, "scheduler.pt"))

        with open(os.path.join(epoch_dir, "state.json"), "w") as f:
            json.dump({"task_id": task_id, "epoch": epoch}, f)

        # Remove previous epoch checkpoint for this task
        for name in os.listdir(self._save_dir):
            if name.startswith(f"task_{task_id:02d}_epoch_") and name != f"task_{task_id:02d}_epoch_{epoch:03d}":
                shutil.rmtree(os.path.join(self._save_dir, name), ignore_errors=True)

    def load_latest(self):
        """Load the latest completed task checkpoint.

        Returns:
            dict with keys: task_id, model, optimizer, scheduler, known_classes,
            total_classes, data_memory, metrics, wandb_run_id, config
            Returns None if no checkpoint found.
        """
        latest_link = os.path.join(self._save_dir, "latest")
        if not os.path.islink(latest_link):
            # Fallback: find the highest task_xx directory
            task_dirs = [
                d for d in os.listdir(self._save_dir)
                if d.startswith("task_") and d.count("_") == 1
                and os.path.isdir(os.path.join(self._save_dir, d))
            ]
            if not task_dirs:
                return None
            task_dirs.sort(reverse=True)
            task_dir = os.path.join(self._save_dir, task_dirs[0])
        else:
            task_dir = os.path.join(self._save_dir, os.readlink(latest_link))
            if not os.path.isdir(task_dir):
                # symlink broken, try fallback
                task_dirs = [
                    d for d in os.listdir(self._save_dir)
                    if d.startswith("task_") and d.count("_") == 1
                    and os.path.isdir(os.path.join(self._save_dir, d))
                ]
                if not task_dirs:
                    return None
                task_dirs.sort(reverse=True)
                task_dir = os.path.join(self._save_dir, task_dirs[0])

        return self._load_from_dir(task_dir)

    def load_task(self, task_id):
        """Load a specific task checkpoint."""
        task_dir = os.path.join(self._save_dir, f"task_{task_id:02d}")
        if not os.path.isdir(task_dir):
            return None
        return self._load_from_dir(task_dir)

    def load_epoch_state(self, task_id, epoch):
        """Load an intra-task epoch checkpoint."""
        epoch_dir = os.path.join(self._save_dir, f"task_{task_id:02d}_epoch_{epoch:03d}")
        if not os.path.isdir(epoch_dir):
            return None

        state = {}
        with open(os.path.join(epoch_dir, "state.json")) as f:
            meta = json.load(f)
            state.update(meta)

        model_path = os.path.join(epoch_dir, "model.pt")
        if os.path.exists(model_path):
            state["model"] = torch.load(model_path, map_location="cpu", weights_only=True)

        opt_path = os.path.join(epoch_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            state["optimizer"] = torch.load(opt_path, map_location="cpu", weights_only=True)

        sched_path = os.path.join(epoch_dir, "scheduler.pt")
        if os.path.exists(sched_path):
            state["scheduler"] = torch.load(sched_path, map_location="cpu", weights_only=True)

        return state

    @staticmethod
    def _load_from_dir(task_dir):
        state = {"data_memory": None, "extra": None}
        metadata_path = os.path.join(task_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                state.update(json.load(f))

        model_path = os.path.join(task_dir, "model.pt")
        if os.path.exists(model_path):
            state["model"] = torch.load(model_path, map_location="cpu", weights_only=True)

        opt_path = os.path.join(task_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            state["optimizer"] = torch.load(opt_path, map_location="cpu", weights_only=True)

        sched_path = os.path.join(task_dir, "scheduler.pt")
        if os.path.exists(sched_path):
            state["scheduler"] = torch.load(sched_path, map_location="cpu", weights_only=True)

        mem_path = os.path.join(task_dir, "memory.npz")
        if os.path.exists(mem_path):
            mem = np.load(mem_path, allow_pickle=True)
            state["data_memory"] = (mem["data"], mem["targets"])

        extra_path = os.path.join(task_dir, "extra.pt")
        if os.path.exists(extra_path):
            state["extra"] = torch.load(extra_path, map_location="cpu")

        return state

    def list_checkpoints(self):
        """List all available task checkpoints."""
        ckpts = []
        for name in os.listdir(self._save_dir):
            if name.startswith("task_") and name.count("_") == 1:
                ckpt_path = os.path.join(self._save_dir, name)
                if os.path.isdir(ckpt_path):
                    meta_path = os.path.join(ckpt_path, "metadata.json")
                    if os.path.exists(meta_path):
                        with open(meta_path) as f:
                            meta = json.load(f)
                        ckpts.append({
                            "task_id": meta.get("task_id"),
                            "timestamp": meta.get("timestamp"),
                            "dir": ckpt_path,
                        })
        ckpts.sort(key=lambda x: x["task_id"])
        return ckpts

    @staticmethod
    def resume_args(checkpoint_dir, args_dict):
        """Update args dict for resume. Restores metrics so trainer continues correctly.

        Args:
            checkpoint_dir: path to checkpoint directory
            args_dict: the original args dict, will be mutated in-place

        Returns:
            (start_task, restored_metrics, wandb_run_id)
        """
        ckpt = CheckpointManager(checkpoint_dir)
        state = ckpt.load_latest()
        if state is None:
            warnings.warn(f"No checkpoint found in {checkpoint_dir}, starting from scratch.")
            return 0, None, None

        args_dict["resume"] = True
        args_dict["checkpoint_dir"] = checkpoint_dir
        return state.get("task_id", -1) + 1, state.get("metrics"), state.get("wandb_run_id")


def _json_default(obj):
    """Handle non-serializable types in json."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, (torch.Tensor,)):
        return obj.tolist()
    return str(obj)


def _sanitize_config(config):
    """Convert config dict to JSON-safe form (handles torch.device, numpy, etc. recursively)."""
    if isinstance(config, dict):
        return {k: _sanitize_config(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        return [_sanitize_config(v) for v in config]
    if isinstance(config, (np.integer,)):
        return int(config)
    if isinstance(config, (np.floating,)):
        return float(config)
    if isinstance(config, np.ndarray):
        return config.tolist()
    if isinstance(config, torch.device):
        return str(config)
    return config
