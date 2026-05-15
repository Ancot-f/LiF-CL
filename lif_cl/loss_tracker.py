from collections import defaultdict


class LossTracker:
    """Per-epoch loss tracking with monotonic wandb step.

    Uses an internal monotonic step counter to avoid step collision
    when multiple phases (func/rd) train within a single task.

    Usage:
        tracker = LossTracker(wandb_logger, task_id=0)

        # In training loop, per batch:
        tracker.update(ce=2.3, kd=0.5, prompt=0.01)

        # At end of epoch:
        tracker.flush(epoch=0)           # step 0: loss/ce, loss/kd
        tracker.flush(epoch=1)           # step 1: loss/ce, loss/kd
    """

    _global_step = 0  # monotonic across all trackers

    def __init__(self, wandb_logger, task_id=0):
        self._wandb = wandb_logger
        self._task_id = task_id
        self._accum = defaultdict(float)
        self._weights = defaultdict(float)

    def update(self, batch_size=1, **named_losses):
        """Accumulate batch losses with batch_size weighting.

        Args:
            batch_size: size of current batch (for weighted averaging)
            **named_losses: keyword arguments where key=loss_name, value=loss_scalar
        """
        for name, value in named_losses.items():
            self._accum[name] += value * batch_size
            self._weights[name] += batch_size

    def _compute_averages(self):
        averages = {}
        for name in self._accum:
            if self._weights[name] > 0:
                averages[name] = self._accum[name] / self._weights[name]
        return averages

    def flush(self, epoch):
        """Compute epoch averages, log to wandb (auto-increment step), reset accumulators.

        Returns:
            dict: {loss_name: average_value} for local logging
        """
        averages = self._compute_averages()

        if self._wandb is not None and self._wandb.initialized:
            log_data = {}
            for name, avg in averages.items():
                log_data[f"loss/{name}"] = avg
            log_data["epoch"] = epoch
            # 不指定 step，让 wandb 自动递增，避免与 main 循环的 step 冲突
            self._wandb.log_metrics(log_data)

        self._accum.clear()
        self._weights.clear()
        return averages

    @classmethod
    def reset_global_step(cls):
        """Reset the global step counter (called at start of new run)."""
        cls._global_step = 0

    def reset(self):
        """Reset accumulators without logging."""
        self._accum.clear()
        self._weights.clear()

    @property
    def task_id(self):
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        self._task_id = value
