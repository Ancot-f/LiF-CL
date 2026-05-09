"""Centralized path resolution for LiF-CL framework.

All sub-projects use this module to locate datasets and pre-trained models,
instead of hardcoding paths.

The root is auto-detected as the parent of this lif_cl/ directory.
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_root():
    return _ROOT


def get_dataset_path(dataset_name=None):
    """Get path to dataset directory or specific dataset.

    Args:
        dataset_name: optional, e.g. 'cifar100', 'imagenet-r', 'vtab'
    Returns:
        str: absolute path
    """
    base = os.path.join(_ROOT, "dataset")
    if dataset_name is None:
        return base
    return os.path.join(base, dataset_name)


def get_premodel_path(model_name=None):
    """Get path to pre-trained model directory or specific model file.

    Args:
        model_name: optional, e.g. 'model.safetensors'
    Returns:
        str: absolute path
    """
    base = os.path.join(_ROOT, "pre-model")
    if model_name is None:
        return base
    return os.path.join(base, model_name)


_SEMA_DATASET_MAP = {
    "cifar10": get_dataset_path(),
    "cifar100": get_dataset_path(),
    "imagenet_r": get_dataset_path("imagenet-r"),
    "imagenet_a": get_dataset_path("imagenet-a"),
    "vtab": get_dataset_path("vtab"),
    "cub": get_dataset_path("cub"),
    "domainnet": get_dataset_path("domainnet"),
    "omnibenchmark": get_dataset_path("omnibenchmark"),
    "objectnet": get_dataset_path("objectnet"),
}


def resolve_data_path(dataset_name):
    """Resolve dataset path from dataset name.

    Used by data.py download_data() to get the correct path.
    """
    name = dataset_name.lower().replace("-", "_")
    if name in _SEMA_DATASET_MAP:
        return _SEMA_DATASET_MAP[name]
    return get_dataset_path(name.replace("_", "-"))
