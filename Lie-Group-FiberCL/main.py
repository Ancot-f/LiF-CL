import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lif_cl.config import ConfigManager
from lif_cl.seed import set_seed, set_device
from lif_cl.wandb_logger import WandbLogger


def main():
    parser = argparse.ArgumentParser(description="Lie-Group FiberCL")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*", default=[],
                        help="Override config values, e.g. lr=0.01 seed=[99]")
    args = parser.parse_args()

    # Load config
    overrides = {}
    for override in args.overrides:
        if "=" in override:
            key, val = override.split("=", 1)
            overrides[key] = val
    cfg = ConfigManager(args.config, cli_overrides=overrides if overrides else None)
    params = cfg.to_dict()

    # Setup
    set_seed(params["seed"][0])
    devices = set_device(params["device"])
    params["device"] = devices

    # wandb
    wandb_cfg = params.get("wandb", {})
    wandb_logger = WandbLogger(
        project=wandb_cfg.get("project", "LiF-CL"),
        group="Lie-Group",
        config=params,
        tags=wandb_cfg.get("tags", []),
    )

    # Logging
    logdir = f"logs/{params['model_name']}/{params['dataset']}/{params['init_cls']}_{params['increment']}"
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, f"seed_{params['seed'][0]}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("Config: %s", params)

    # TODO: DataManager, ModelRegistry.get, training loop with LossTracker
    logging.info("Lie-Group FiberCL initialized. Training loop to be implemented.")

    wandb_logger.finish()


if __name__ == "__main__":
    main()
