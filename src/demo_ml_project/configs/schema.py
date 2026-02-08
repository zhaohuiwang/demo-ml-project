

# project-root(demo-ml-project)/demo_ml_project.configs.schema.py

from enum import Enum
from pathlib import Path
from typing import List, Tuple, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)

class Device(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"

class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_data_path: str = Field(..., description="Path to parquet file")
    drop_columns: List[str] = Field(default_factory=list)
    cat_cols: List[str] = Field(default_factory=list)
    date_cols: List[str] = Field(default_factory=list)
    num_cols: List[str] = Field(default_factory=list)
    target_cols: List[str] = Field(..., min_length=1)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_size: float = Field(0.2, gt=0.0, lt=1.0)
    random_state: int = 42
    batch_size: int = Field(64, ge=4)
    max_epochs: int = Field(100, ge=10)
    patience: int = Field(10, ge=3)


class OptunaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_trials: int = Field(15, ge=1)
    n_epochs_per_trial: int = Field(5, ge=1)

    layer_range: Tuple[int, int] = (1, 4)
    units_list: List[int] = Field(default_factory=lambda: [16, 64, 128, 256])
    dropout_range: Tuple[float, float] = (0.1, 0.4)
    lr_range: Tuple[float, float] = (1e-4, 1e-2)

    sampler: Literal["tpe", "random", "cmaes"] = "tpe"
    pruner: Literal["median", "hyperband", "nop"] = "median"

    @field_validator("layer_range", "dropout_range", "lr_range")
    @classmethod
    def check_range(cls, v: Tuple):
        lo, hi = v
        if lo >= hi:
            raise ValueError(f"Lower bound must be < upper bound (got {lo} ≥ {hi})")
        return v


class ExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: str = "model_export"
    weights: str = "champion_weights.pth"
    in_scaler: str = "input_scaler.pkl"
    tar_scaler: str = "target_scaler.pkl"
    cat_encoder: str = "categorical_input_encoders.pkl"
    metadata: str = "metadata.json"


class RootConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    training: TrainingConfig
    optuna: OptunaConfig
    export: ExportConfig

    project_root: Optional[Path] = None
    seed: int = 42
    device: Device = Device.CUDA

    @model_validator(mode="after")
    def resolve_paths(self):
        if self.project_root is None:
            raise ValueError("project_root must be set (usually via hydra)")
        root = self.project_root

        if not Path(self.data.train_data_path).is_absolute():
            self.data.train_data_path = str(root / self.data.train_data_path)

        if not Path(self.export.dir).is_absolute():
            self.export.dir = str(root / self.export.dir)

        return self