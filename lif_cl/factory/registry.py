import importlib
import logging


class ModelRegistry:
    """Decorator-based model registry with lazy imports.

    Usage:
        # Register with decorator
        @ModelRegistry.register("inflora")
        class InfLoRA(BaseLearner):
            ...

        # Register with explicit module path (lazy import)
        ModelRegistry.register("sema", module="models.sema")

        # Get model instance
        model = ModelRegistry.get("inflora", args)
    """

    _models = {}

    @classmethod
    def register(cls, name, module=None):
        """Register a model class.

        Can be used as decorator or with explicit module path for lazy imports.

        Args:
            name: string identifier for the model
            module: optional dotted path for lazy import (e.g. 'models.inflora')
        """
        def decorator(model_cls):
            cls._models[name.lower()] = {"cls": model_cls, "module": module, "lazy": module is not None}
            logging.debug("Registered model: %s -> %s", name, model_cls.__name__)
            return model_cls

        if module is not None:
            cls._models[name.lower()] = {"cls": None, "module": module, "lazy": True}
            return None

        # Being used as decorator @ModelRegistry.register("name")
        return decorator

    @classmethod
    def get(cls, name, args):
        """Get a model instance by name.

        Lazy-imports the module on first access.
        """
        name = name.lower()
        if name not in cls._models:
            raise KeyError(
                f"Unknown model '{name}'. Available: {list(cls._models.keys())}"
            )

        entry = cls._models[name]

        if entry["lazy"] and entry["cls"] is None:
            module = importlib.import_module(entry["module"])
            # Find the Learner class in the module by convention
            entry["cls"] = getattr(module, "Learner")

        return entry["cls"](args)

    @classmethod
    def list_models(cls):
        return list(cls._models.keys())

    @classmethod
    def has(cls, name):
        return name.lower() in cls._models
