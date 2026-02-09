

# project-root(demo-ml-project)/demo_ml_project.configs.schema.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
from enum import Enum

from hydra.core.config_store import ConfigStore


class Device(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"


@dataclass
class DataConfig:
    train_data_path: str
    drop_columns: List[str] = field(default_factory=list)
    cat_cols: List[str] = field(default_factory=list)
    date_cols: List[str] = field(default_factory=list)
    num_cols: List[str] = field(default_factory=list)
    target_cols: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Manual validation / coercion
        self.train_data_path = str(Path(self.train_data_path).resolve())


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10

    def __post_init__(self):
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        

@dataclass
class OptunaConfig:
    n_trials: int = 15
    n_epochs_per_trial: int = 5
    layer_range: Tuple[int, int] = (1, 4)
    units_list: List[int] = field(default_factory=lambda: [16, 64, 128, 256])
    dropout_range: Tuple[float, float] = (0.1, 0.4)
    lr_range: Tuple[float, float] = (0.0001, 0.01)
    
    sampler: str = "tpe"
    pruner:  str = "median"

    def __post_init__(self):
        lo, hi = self.layer_range
        if lo >= hi or lo < 1:
            raise ValueError(f"Invalid layer_range: {self.layer_range}")
        lo_d, hi_d = self.dropout_range
        if lo_d >= hi_d or lo_d < 0 or hi_d > 1:
            raise ValueError(f"Invalid dropout_range: {self.dropout_range}")
        lo_lr, hi_lr = self.lr_range
        if lo_lr >= hi_lr or lo_lr <= 0:
            raise ValueError(f"Invalid lr_range: {self.lr_range}")
        
        # Similar checks for dropout_range, lr_range, etc.
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")



@dataclass
class ExportConfig:
    dir: str = "model_export"
    weights: str = "champion_weights.pth"
    in_scaler: str = "input_scaler.pkl"
    tar_scaler: str = "target_scaler.pkl"
    cat_encoder: str = "categorical_input_encoders.pkl"
    metadata: str = "metadata.json"


@dataclass
class RootConfig:
    data: DataConfig
    training: TrainingConfig
    optuna: OptunaConfig
    export: ExportConfig

    project_root: Path | None = None
    seed: int = 42
    device: Device = Device.CUDA

    def __post_init__(self):
        if self.project_root is None:
            raise ValueError("project_root must be set")
        root = self.project_root

        # Path resolution (like your old @model_validator)
        if not Path(self.data.train_data_path).is_absolute():
            self.data.train_data_path = str(root / self.data.train_data_path)

        if not Path(self.export.dir).is_absolute():
            self.export.dir = str(root / self.export.dir)

# Register
cs = ConfigStore.instance()
cs.store(name="config", node=RootConfig)
cs.store(group="data", name="default", node=DataConfig)
cs.store(group="training", name="default", node=TrainingConfig)
cs.store(group="optuna", name="default", node=OptunaConfig)
cs.store(group="export", name="default", node=ExportConfig)