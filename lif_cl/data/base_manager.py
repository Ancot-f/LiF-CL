import logging
import numpy as np


class BaseDataManager:
    """Base class for continual learning data management.

    Handles class ordering, task splitting, and index mapping.
    Sub-projects override _setup_data() for dataset-specific loading.

    Args:
        dataset_name: str, e.g. 'cifar100', 'imagenet_r'
        shuffle: bool, whether to shuffle class order
        seed: int, random seed for shuffling
        init_cls: int, number of classes in first task
        increment: int, classes per subsequent task
        args: dict, additional config
    """

    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args=None):
        self.args = args
        self.dataset_name = dataset_name
        self._increments = []
        self._class_order = []
        self._setup_data(dataset_name, shuffle, seed)
        self._compute_increments(init_cls, increment)

    def _setup_data(self, dataset_name, shuffle, seed):
        """Override in subclass to load data and set:
        - self._train_data, self._train_targets
        - self._test_data, self._test_targets
        - self._train_trsf, self._test_trsf, self._common_trsf
        - self.use_path
        """
        raise NotImplementedError

    def _compute_increments(self, init_cls, increment):
        assert init_cls <= len(self._class_order), "Not enough classes for init_cls."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    @property
    def nb_classes(self):
        return len(self._class_order)

    @property
    def class_order(self):
        return self._class_order

    def get_task_size(self, task):
        return self._increments[task]

    def _setup_class_order(self, num_classes, shuffle, seed):
        order = list(range(num_classes))
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(num_classes).tolist()
        self._class_order = order
        logging.info("Class order: %s", self._class_order)

    @staticmethod
    def _map_new_class_index(y, order):
        return np.array([order.index(cls) for cls in y])

    @staticmethod
    def _select(x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]
