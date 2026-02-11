
# src/demo_ml_project/configs/__init__.py (or put at bottom of schema.py)

from hydra_zen import store

from .schema import (
    RootConfig,
    DataConfig,
    TrainingConfig,
    OptunaConfig,
    ExportConfig,
)

store(RootConfig, name="config", group=None)          # root
store(DataConfig, name="default", group="data")
store(TrainingConfig, name="default", group="training")
store(OptunaConfig, name="default", group="optuna")
store(ExportConfig, name="default", group="export")

# Optional: call this somewhere early (e.g. in train.py)
# store.add_to_hydra_store()