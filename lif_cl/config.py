import json
import argparse
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


class ConfigManager:
    """Unified config loader supporting YAML (primary) and JSON (compat).

    Usage:
        # YAML config
        cfg = ConfigManager("configs/experiment.yaml")

        # YAML with CLI overrides
        cfg = ConfigManager("configs/experiment.yaml", cli_overrides={"lr": 0.01, "seed": [99]})

        # JSON config (backwards compat)
        cfg = ConfigManager("configs/experiment.json")

        # As dict (compatible with existing sub-projects)
        args_dict = cfg.to_dict()

        # As argparse.Namespace
        args_ns = cfg.to_namespace()
    """

    def __init__(self, config_path=None, cli_overrides=None):
        self._config = {}
        self._config_path = config_path

        if config_path is not None:
            self._config = self._load_file(config_path)

        if cli_overrides:
            self._apply_overrides(cli_overrides)

    def _load_file(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        if path.suffix in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError("pyyaml is required for YAML config files")
            with open(path) as f:
                return yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path) as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    def _apply_overrides(self, overrides):
        for key, value in overrides.items():
            keys = key.split(".")
            d = self._config
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = self._parse_value(value)

    @staticmethod
    def _parse_value(value):
        if isinstance(value, str):
            if value.lower() == "true":
                return True
            if value.lower() == "false":
                return False
            if value.lower() == "none":
                return None
            try:
                return int(value)
            except ValueError:
                pass
            try:
                return float(value)
            except ValueError:
                pass
        return value

    def to_dict(self):
        return self._config.copy()

    def to_namespace(self):
        return argparse.Namespace(**self._flat_dict())

    def _flat_dict(self):
        result = {}
        for key, value in self._config.items():
            if isinstance(value, dict) and not any(
                isinstance(v, (dict, list)) for v in value.values()
            ):
                # Flatten one level: wandb.project -> wandb_project
                for sub_key, sub_val in value.items():
                    result[f"{key}_{sub_key}"] = sub_val
            else:
                result[key] = value
        return result

    def get(self, key, default=None):
        keys = key.split(".")
        d = self._config
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    def __getitem__(self, key):
        return self._config[key]

    def __contains__(self, key):
        return key in self._config

    def __repr__(self):
        return f"ConfigManager({self._config})"

    def save(self, output_path):
        path = Path(output_path)
        with open(path, "w") as f:
            if path.suffix in (".yaml", ".yml"):
                if yaml is None:
                    raise ImportError("pyyaml is required to save YAML")
                yaml.safe_dump(self._config, f, default_flow_style=False)
            else:
                json.dump(self._config, f, indent=2)
