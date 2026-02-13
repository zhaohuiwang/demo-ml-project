

# project-root(demo-ml-project)/demo_ml_project.configs.schema.py

from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)

class GlobalConfig(BaseModel):
    project_root: Optional[Path] = None
    seed: int = 42
    device: Literal["cuda", "cpu", "mps"] = "cuda"

class DataConfig(BaseModel):
    """Configuration for input data sources and feature groups."""

    train_data_path: Path = Field(..., description="Path to the training Parquet/CSV file")
    drop_columns: List[str] = Field(default_factory=list, description="Columns to drop during loading")
    cat_cols: List[str] = Field(default_factory=list, description="Categorical feature column names")
    date_cols: List[str] = Field(default_factory=list, description="Date/datetime columns (not yet used)")
    num_cols: List[str] = Field(default_factory=list, description="Numerical feature column names")
    target_cols: List[str] = Field(default_factory=list, description="Target column names (multi-target regression)")

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    @field_validator("train_data_path", mode="after")
    @classmethod
    def check_file_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Data file not found: {v}")
        return v


class TrainingConfig(BaseModel):
    """Hyperparameters related to training process and splitting."""

    test_size: float = Field(0.2, gt=0.0, lt=1.0, description="Fraction of data for validation")
    random_state: int = Field(42, description="Random seed for reproducibility")
    batch_size: int = Field(64, ge=1, description="Batch size for DataLoader")
    max_epochs: int = Field(200, ge=1, description="Maximum epochs for final training")
    patience: int = Field(12, ge=1, description="Early stopping patience")

    cv: "CVConfig" = Field(
        default_factory=lambda: CVConfig(),
        description="Cross-validation settings used during HPO"
    )

    model_config = ConfigDict(extra="forbid")


class CVConfig(BaseModel):
    """Cross-validation settings for more robust hyperparameter optimization."""

    enabled: bool = Field(False, description="Enable k-fold / time-series CV in Optuna objective")
    n_folds: int = Field(5, ge=2, le=10, description="Number of folds (higher = more stable, slower)")
    strategy: Literal["kfold", "timeseries"] = Field(
        "kfold", description="Splitter type: kfold or timeseries"
    )
    shuffle: bool = Field(True, description="Shuffle before splitting (ignored for timeseries)")
    random_state: Optional[int] = Field(42, description="Seed for reproducible splits")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def enforce_timeseries_rules(self) -> "CVConfig":
        if self.strategy == "timeseries" and self.shuffle:
            self.shuffle = False  # timeseries split should never shuffle
        return self


class OptunaConfig(BaseModel):
    """Hyperparameter optimization settings using Optuna."""

    n_trials: int = Field(30, ge=1, description="Number of Optuna trials")
    n_epochs_per_trial: int = Field(8, ge=1, description="Epochs per trial (short for speed)")
    layer_range: Tuple[int, int] = Field((1, 5), description="Min/max number of hidden layers")
    units_list: List[int] = Field(default_factory=lambda: [32, 64, 96, 128, 192, 256], description="Possible hidden units per layer")
    dropout_range: Tuple[float, float] = Field((0.05, 0.45), description="Dropout probability range")
    lr_range: Tuple[float, float] = Field((1e-5, 5e-2), description="Learning rate range (log scale)")

    sampler: Literal["tpe", "random", "grid"] = Field("tpe")
    pruner: Literal["median", "hyperband", "none"] = Field("hyperband")

    model_config = ConfigDict(extra="forbid")

    @field_validator("layer_range")
    @classmethod
    def check_layer_range(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        lo, hi = v
        if lo >= hi or lo < 1:
            raise ValueError("layer_range must satisfy 1 <= lo < hi")
        return v

    # similar validators for dropout_range, lr_range can be added if desired


class ExportConfig(BaseModel):
    dir: Path = Field(..., description="Base export directory")
    weights: str = Field("model_state.pth", description="Model state dict filename")
    input_scaler: str = Field("input_scaler.joblib", description="Input/feature scaler")
    target_scaler: str = Field("target_scaler.joblib", description="Target scaler")
    cat_encoder: str = Field("cat_encoder.joblib", description="Categorical encoder")
    metadata: str = Field("metadata.json", description="Metadata JSON")

    model_config = {
        "extra": "forbid",          # keep this strict → helps catch typos
        "frozen": False,
    }

    @field_validator("dir", mode="before")
    @classmethod
    def resolve_export_dir(cls, v: str | Path) -> Path:
        return Path(v).resolve()

class MlflowConfig(BaseModel):
    tracking_uri: str = Field(
        "http://127.0.0.1:5000",
        description="MLflow tracking server URI (http://... or file:...)"
    )
    experiment_name: str = Field(
        "demo-ml-tabular-regression",
        description="MLflow experiment name"
    )
    registered_model_name: str = Field("TabularMultiTargetRegressor")
    model_artifact_path: str = Field("model")
    
class RootConfig(BaseModel):
    """Top-level configuration combining all sections."""

    global_: GlobalConfig = Field(..., alias="global")   # note alias to avoid keyword conflict
    data: DataConfig
    training: TrainingConfig
    optuna: OptunaConfig
    export: ExportConfig
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def resolve_paths_and_interpolations(self) -> "RootConfig":
        root = self.global_.project_root
        if root is None:
            raise ValueError("project_root required")

        # Handle Hydra-style ${now:...} in export.dir
        export_dir_str = str(self.export.dir)
        if "${now:%Y%m%d_%H%M%S}" in export_dir_str:
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir_str = export_dir_str.replace("${now:%Y%m%d_%H%M%S}", now_str)
            self.export.dir = Path(export_dir_str)

        # Make paths absolute
        if not self.data.train_data_path.is_absolute():
            self.data.train_data_path = root / self.data.train_data_path

        if not self.export.dir.is_absolute():
            self.export.dir = root / self.export.dir

        return self