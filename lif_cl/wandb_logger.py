import os
import warnings

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False
    wandb = None


class WandbLogger:
    """Unified wandb logger for LiF-CL framework.

    Usage:
        logger = WandbLogger(group="Lie-Group", config=args_dict, tags=["dev"])
        logger.log_metrics({"acc/top1_avg": 0.75}, step=0)
        logger.log_task_matrix(matrix_data, task_id=3)
        logger.log_summary({"avg_acc": 0.82, "bwt": -0.03})
        logger.finish()
    """

    def __init__(
        self,
        project="LiF-CL",
        group=None,
        name=None,
        config=None,
        tags=None,
        notes=None,
        offline=False,
        resume_id=None,
    ):
        self._project = project
        self._group = group
        self._initialized = False
        self._run = None

        if not _HAS_WANDB:
            warnings.warn("wandb not installed. WandbLogger will be a no-op.")
            return

        if offline or os.environ.get("WANDB_MODE") == "offline":
            os.environ["WANDB_MODE"] = "offline"

        init_kwargs = dict(
            project=project,
            group=group,
            name=name,
            config=config or {},
            tags=tags or [],
            notes=notes,
        )
        if resume_id:
            init_kwargs["id"] = resume_id
            init_kwargs["resume"] = "allow"
        else:
            init_kwargs["reinit"] = True

        self._run = wandb.init(**init_kwargs)
        self._initialized = True
        self._define_default_metrics()

    @property
    def run(self):
        return self._run

    @property
    def initialized(self):
        return self._initialized

    def _define_default_metrics(self):
        """Set up paper-style CL experiment charts.

        Organizes metrics into sections (eval/, loss/, expansion/) and sets
        appropriate summaries for each.
        """
        # eval — per-task accuracy, final = last task average, last value
        for metric in ["eval/acc_avg", "eval/acc_old", "eval/acc_new",
                       "eval/cnn_top1", "eval/cnn_top5",
                       "eval/nme_top1", "eval/nme_top5",
                       "eval/forgetting"]:
            self._run.define_metric(metric, summary="last")

        # loss — per-epoch, summary = minimum across all epochs
        for metric in ["loss/func", "loss/rd", "loss/ce", "loss/kd",
                       "loss/fe", "loss/aux"]:
            self._run.define_metric(metric, summary="min")

        # expansion — per-task, summary = last
        for metric in ["expansion/total_params", "expansion/trainable_params",
                       "expansion/param_ratio"]:
            self._run.define_metric(metric, summary="last")

    def define_metric(self, metric, step_metric=None, summary=None):
        """Define a wandb metric with optional x-axis and summary.

        Args:
            metric: metric name (e.g., "eval/cnn_top1")
            step_metric: metric to use as x-axis (e.g., "eval/total_classes")
            summary: summary type ("last", "min", "max", None)
        """
        if not self._initialized:
            return
        kwargs = {}
        if step_metric is not None:
            kwargs["step_metric"] = step_metric
        if summary is not None:
            kwargs["summary"] = summary
        self._run.define_metric(metric, **kwargs)

    def log_metrics(self, metrics, step=None):
        """Log scalar metrics to wandb."""
        if not self._initialized:
            return
        self._run.log(metrics, step=step)

    def log_losses(self, losses, step=None, epoch=None):
        """Log per-epoch loss averages.

        Args:
            losses: dict of {loss_name: average_value}
            step: global step (task_id)
            epoch: current epoch within the task
        """
        if not self._initialized:
            return
        log_data = {}
        for name, value in losses.items():
            log_data[f"loss/{name}"] = value
        if epoch is not None:
            log_data["epoch"] = epoch
        self._run.log(log_data, step=step)

    def log_accuracy(self, acc_dict, step=None):
        """Log accuracy metrics.

        Args:
            acc_dict: dict with keys like 'top1', 'top5', 'old', 'new', 'grouped_XX-YY'
            step: global step (task_id)
        """
        if not self._initialized:
            return
        log_data = {}
        for key, value in acc_dict.items():
            log_data[f"acc/{key}"] = value
        self._run.log(log_data, step=step)

    def log_task_matrix(self, matrix):
        """Log final per-task accuracy matrix.

        Logged once at end of training. Values are numeric so wandb can render
        them as a color-coded heatmap table.

        Args:
            matrix: list of lists, where matrix[i][j] = accuracy on task j after training task i.
        """
        if not self._initialized or not matrix:
            return
        num_cols = len(matrix[-1])
        columns = [f"Task_{j}" for j in range(num_cols)]
        rows = []
        for i, row in enumerate(matrix):
            padded = list(row) + [None] * (num_cols - len(row))
            row_data = [f"After_T{i}"] + [
                round(float(v), 2) if v is not None else None for v in padded
            ]
            rows.append(row_data)

        table = wandb.Table(data=rows, columns=[" "] + columns)
        self._run.log({"task_matrix": table})

    def log_summary(self, summary):
        """Write to wandb run summary."""
        if not self._initialized:
            return
        for key, value in summary.items():
            self._run.summary[key] = value

    def log_config(self, config):
        """Update wandb config (can be called after init to add extra params)."""
        if not self._initialized:
            return
        self._run.config.update(config, allow_val_change=True)

    def finish(self):
        if self._initialized:
            self._run.finish()
            self._initialized = False
