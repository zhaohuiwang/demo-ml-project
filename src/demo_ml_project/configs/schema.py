

# project-root(demo-ml-project)/demo_ml_project.configs.schema.py

from pathlib import Path
from typing import List, Tuple, Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ConfigDict,
    model_validator,
)

# from enum import Enum
# class Device(str, Enum):
#     CUDA = "cuda"
#     CPU = "cpu"
#     MPS = "mps"

class DataConfig(BaseModel):
    train_data_path: Path = Field(..., description="Path to training data CSV/Parquet/etc.")
    drop_columns: List[str] = Field(default_factory=list)
    cat_cols: List[str] = Field(default_factory=list)
    date_cols: List[str] = Field(default_factory=list)
    num_cols: List[str] = Field(default_factory=list)
    target_cols: List[str] = Field(default_factory=list)

    model_config = ConfigDict(
        extra="forbid",                 # catch unknown keys
        coerce_numbers_to_str=False,
    )

    @field_validator("train_data_path", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        return Path(v).resolve()


class TrainingConfig(BaseModel):
    test_size: float = Field(0.2, gt=0.0, lt=1.0)
    random_state: int = 42
    batch_size: int = Field(64, ge=1)
    max_epochs: int = Field(100, ge=1)
    patience: int = Field(10, ge=1)

    model_config = ConfigDict(extra="forbid")

        
class OptunaConfig(BaseModel):
    n_trials: int = Field(15, ge=1)
    n_epochs_per_trial: int = Field(5, ge=1)
    layer_range: Tuple[int, int] = Field((1, 4), description="min/max hidden layers")
    units_list: List[int] = Field(default_factory=lambda: [16, 64, 128, 256])
    dropout_range: Tuple[float, float] = Field((0.1, 0.4))
    lr_range: Tuple[float, float] = Field((0.0001, 0.01))

    sampler: Literal["tpe", "random", "grid"] = "tpe"          # add more if you use them
    pruner: Literal["median", "hyperband", "none"] = "median"  # adjust as needed

    model_config = ConfigDict(extra="forbid")

    @field_validator("layer_range")
    @classmethod
    def check_layer_range(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        lo, hi = v
        if lo >= hi or lo < 1:
            raise ValueError(f"Invalid layer_range: {v} (lo < hi and lo >= 1)")
        return v

    @field_validator("dropout_range")
    @classmethod
    def check_dropout_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = v
        if lo >= hi or lo < 0 or hi > 1:
            raise ValueError(f"Invalid dropout_range: {v} (0 ≤ lo < hi ≤ 1)")
        return v

    @field_validator("lr_range")
    @classmethod
    def check_lr_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = v
        if lo >= hi or lo <= 0:
            raise ValueError(f"Invalid lr_range: {v} (0 < lo < hi)")
        return v

class ExportConfig(BaseModel):
    dir: Path = Field(default_factory=lambda: Path("model_export"))
    weights: str = "champion_weights.pth"
    in_scaler: str = "input_scaler.pkl"
    tar_scaler: str = "target_scaler.pkl"
    cat_encoder: str = "categorical_input_encoders.pkl"
    metadata: str = "metadata.json"

    model_config = ConfigDict(extra="forbid")

    @field_validator("dir", mode="before")
    @classmethod
    def resolve_export_dir(cls, v: str | Path) -> Path:
        return Path(v).resolve()

class RootConfig(BaseModel):
    data: DataConfig
    training: TrainingConfig
    optuna: OptunaConfig
    export: ExportConfig

    project_root: Path | None = None
    seed: int = 42
    device: Literal["cuda", "cpu", "mps"] = "cuda"

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,      # validate defaults too
    )

    @model_validator(mode="after")
    def resolve_paths_relative_to_root(self) -> "RootConfig":
        if self.project_root is None:
            raise ValueError("project_root must be set")

        root = self.project_root

        # Make data path absolute if relative
        if not self.data.train_data_path.is_absolute():
            self.data.train_data_path = root / self.data.train_data_path

        # Same for export dir
        if not self.export.dir.is_absolute():
            self.export.dir = root / self.export.dir

        return self