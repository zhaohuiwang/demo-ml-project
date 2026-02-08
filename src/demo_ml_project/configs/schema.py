

# project-root(demo-ml-project)/demo_ml_project.configs.schema.py
import optuna
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from typing import Annotated, Any, Literal
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)


@dataclass
class PreprocessingArtifacts:
    processed_df: pd.DataFrame
    cat_encoder: OrdinalEncoder
    num_scaler: StandardScaler
    tar_scaler: StandardScaler

@dataclass
class TrainingArtifacts:
    """Structured container instead of dict[str, Any]"""

    model: nn.Module
    study: optuna.Study
    emb_sizes: list[tuple[int, int]]
    preprocessors: dict[str, Any]
    # metrics: dict[str, float]   # usually added later

# --- CONFIG VALIDATION SCHEMA ---
class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")  # stricter, prevents typos

    train_data_path: Path | None = None
    drop_columns: list[str] = Field(default_factory=list)
    cat_cols: list[str] = Field(default_factory=list)
    date_cols: list[str] = Field(default_factory=list)
    num_cols: list[str] = Field(default_factory=list)
    target_cols: list[str] = Field(..., min_length=1)


    @field_validator("train_data_path", mode="before")
    @classmethod
    def normalize_path(cls, v: str | Path | None) -> Path | None:
        if v is None:
            return None
        return Path(v)

class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_size: Annotated[float, Field(gt=0, lt=1)] = 0.2
    random_state: int = 42
    batch_size: Annotated[int, Field(ge=4, le=8192)] = 256
    max_epochs: Annotated[int, Field(ge=5)] = 300
    patience: Annotated[int, Field(ge=3)] = 12


class OptunaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_trials: Annotated[int, Field(ge=4, description="Number of Optuna trials")] = 40
    n_epochs_per_trial: Annotated[int, Field(ge=5, le=300, description="Epochs per trial")] = 50

    layer_range: Annotated[tuple[int, int], Field(min_length=2, max_length=2, description="Min/max number of layers")] = (1, 5)
    units_list: Annotated[list[int], Field(min_length=1)] = [32, 64, 96, 128, 192, 256, 384, 512]

    dropout_range: Annotated[tuple[float, float], Field(min_length=2, max_length=2, description="Min/max dropout rate")] = (0.0, 0.5)
    lr_range: Annotated[tuple[float, float], Field(min_length=2, max_length=2, description="Min/max learning rate")] = (1e-5, 3e-2)

   
    # Note: Field is appropriate for simple type & bounds checks (ge, le), metadata (description), and required values (...). For custom logic, cross-element validation, and complex rules (lo < hi, positive ranges, etc.), we need @field_validator().

    # Very useful in practice
    pruner: Literal["median", "percentile", "hyperband", "nop"] = "median"
    sampler: Literal["tpe", "random", "optuna.samplers.CmaEsSampler"] = "tpe"

    @field_validator("layer_range", "dropout_range", "lr_range")
    @classmethod
    def validate_range(cls, v: tuple[float | int, float | int]):
        lo, hi = v
        if lo >= hi:
            raise ValueError(f"Lower bound must be < upper bound (got {lo} ≥ {hi})")
        return v

    @field_validator("lr_range")
    @classmethod
    def validate_lr_positive(cls, v):
        if v[0] <= 0:
            raise ValueError("Learning rate must be > 0")
        return v


class ExportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: str = "artifacts"
    weights: str = "model.pt"
    in_scaler: str = "num_scaler.pkl"
    tar_scaler: str = "target_scaler.pkl"
    cat_encoder: str = "cat_encoder.pkl"
    metadata: str = "metadata.json"

class ConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    training: TrainConfig
    optuna: OptunaConfig
    export: ExportConfig

    project_root: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def resolve_paths(self) -> "ConfigSchema":
        if self.project_root is None:
            raise ValueError("project_root must be injected")

        root = self.project_root

        if self.data.train_data_path and not self.data.train_data_path.is_absolute():
            self.data.train_data_path = root / self.data.train_data_path

        if not Path(self.export.dir).is_absolute():
            self.export.dir = str(root / self.export.dir)

        return self


