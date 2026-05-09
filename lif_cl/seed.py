import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(device_spec=None):
    """Parse device spec into list of torch.device.

    Priority: user-specified > auto-detect (max free memory) > CPU fallback.

    Args:
        device_spec:
            - None, "auto", ["auto"] → auto-select GPU with most free memory
            - [-1], "-1" → CPU
            - [0], "0" → cuda:0
            - [0, 1], "0,1" → cuda:0 and cuda:1

    Returns:
        list of torch.device
    """
    # Normalize to list of strings
    if device_spec is None:
        return [_auto_select_gpu()]
    if isinstance(device_spec, str):
        device_spec = [d.strip().lower() for d in device_spec.split(",") if d.strip()]

    devices = []
    for d in device_spec:
        if isinstance(d, str) and d in ("auto",):
            return [_auto_select_gpu()]
        if isinstance(d, str):
            d = int(d)
        if d == -1:
            devices.append(torch.device("cpu"))
        else:
            devices.append(torch.device(f"cuda:{d}"))
    return devices


def _auto_select_gpu():
    """Return the GPU with the most free memory, or CPU if no GPU available."""
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return torch.device("cpu")

    import logging
    best_id = 0
    best_mem = 0
    errors = 0

    for i in range(torch.cuda.device_count()):
        try:
            free_mem = torch.cuda.mem_get_info(i)[0]  # (free, total)
        except RuntimeError as e:
            logging.warning("Cannot query cuda:%d memory: %s", i, e)
            errors += 1
            continue
        if free_mem > best_mem:
            best_mem = free_mem
            best_id = i

    if errors == torch.cuda.device_count():
        logging.warning("All GPUs unavailable for query, falling back to cuda:0")
        return torch.device("cuda:0")

    free_gb = best_mem / (1024 ** 3)
    logging.info("Auto-selected cuda:%d (%.1f GB free)", best_id, free_gb)
    return torch.device(f"cuda:{best_id}")
