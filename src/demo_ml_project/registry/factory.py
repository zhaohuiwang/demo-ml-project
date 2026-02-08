

# Registry Factory

from my_ml_project.registry.local import LocalModelRegistry
from my_ml_project.registry.mlflow import MLflowModelRegistry


def get_model_registry(cfg):
    if cfg.registry.type == "local":
        return LocalModelRegistry(cfg.registry.root_dir)

    if cfg.registry.type == "mlflow":
        return MLflowModelRegistry()

    raise ValueError(...)
