

# project-root(demo-ml-project)/demo_ml_project.configs.schema.py

from pathlib import Path
from typing import List, Tuple, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)


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

    @field_validator("train_data_path", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        return Path(v).resolve()


class TrainingConfig(BaseModel):
    """Hyperparameters related to training process and splitting."""

    test_size: float = Field(0.2, gt=0.0, lt=1.0, description="Fraction of data for validation")
    random_state: int = Field(42, description="Random seed for reproducibility")
    batch_size: int = Field(64, ge=1, description="Batch size for DataLoader")
    max_epochs: int = Field(100, ge=1, description="Maximum epochs for final training")
    patience: int = Field(10, ge=1, description="Early stopping patience")

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

    n_trials: int = Field(15, ge=1, description="Number of Optuna trials")
    n_epochs_per_trial: int = Field(5, ge=1, description="Epochs per trial (short for speed)")
    layer_range: Tuple[int, int] = Field((1, 4), description="Min/max number of hidden layers")
    units_list: List[int] = Field(default_factory=lambda: [16, 64, 128, 256], description="Possible hidden units per layer")
    dropout_range: Tuple[float, float] = Field((0.1, 0.4), description="Dropout probability range")
    lr_range: Tuple[float, float] = Field((0.0001, 0.01), description="Learning rate range (log scale)")

    sampler: Literal["tpe", "random", "grid"] = Field("tpe")
    pruner: Literal["median", "hyperband", "none"] = Field("median")

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
    """Paths and filenames for model export artifacts."""

    dir: Path = Field(default_factory=lambda: Path("model_export"), description="Export directory")
    weights: str = Field("final_model_weights.pth", description="PyTorch state dict filename")
    in_scaler: str = Field("num_scaler.pkl", description="Numerical features scaler")
    tar_scaler: str = Field("target_scaler.pkl", description="Target scaler")
    cat_encoder: str = Field("cat_encoder.pkl", description="Categorical ordinal encoder")
    metadata: str = Field("training_metadata.json", description="JSON with hparams, metrics, etc.")

    model_config = ConfigDict(extra="forbid")

    @field_validator("dir", mode="before")
    @classmethod
    def resolve_export_dir(cls, v: str | Path) -> Path:
        return Path(v).resolve()


class RootConfig(BaseModel):
    """Top-level configuration combining all sections."""

    project_root: Optional[Path] = None
    seed: int = Field(42, description="Global random seed")
    device: Literal["cuda", "cpu", "mps"] = Field("cuda")

    data: DataConfig
    training: TrainingConfig
    optuna: OptunaConfig
    export: ExportConfig

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> "RootConfig":
        if self.project_root is None:
            raise ValueError("project_root must be set (usually by hydra)")

        root = self.project_root

        if not self.data.train_data_path.is_absolute():
            self.data.train_data_path = root / self.data.train_data_path

        if not self.export.dir.is_absolute():
            self.export.dir = root / self.export.dir

        return self
    
# from pathlib import Path
# from typing import List, Tuple, Literal

# from pydantic import (
#     BaseModel,
#     Field,
#     field_validator,
#     ConfigDict,
#     model_validator,
# )

# # from enum import Enum
# # class Device(str, Enum):
# #     CUDA = "cuda"
# #     CPU = "cpu"
# #     MPS = "mps"

# class DataConfig(BaseModel):
#     train_data_path: Path = Field(..., description="Path to training data CSV/Parquet/etc.")
#     drop_columns: List[str] = Field(default_factory=list)
#     cat_cols: List[str] = Field(default_factory=list)
#     date_cols: List[str] = Field(default_factory=list)
#     num_cols: List[str] = Field(default_factory=list)
#     target_cols: List[str] = Field(default_factory=list)

#     model_config = ConfigDict(
#         extra="forbid",                 # catch unknown keys
#         coerce_numbers_to_str=False,
#     )

#     @field_validator("train_data_path", mode="before")
#     @classmethod
#     def resolve_path(cls, v: str | Path) -> Path:
#         return Path(v).resolve()


# class CVConfig(BaseModel):
#     enabled: bool = Field(
#         default=False,
#         description="Whether to use cross-validation inside Optuna objective"
#     )
#     n_folds: int = Field(
#         default=5,
#         ge=2,
#         le=10,
#         description="Number of CV folds (higher = more stable but slower)"
#     )
#     strategy: Literal["kfold", "timeseries"] = Field(
#         default="kfold",
#         description="Cross-validation splitter type"
#     )
#     shuffle: bool = Field(
#         default=True,
#         description="Whether to shuffle data before splitting (for kfold)"
#     )
#     random_state: int | None = Field(
#         default=42,
#         description="Random seed for reproducibility"
#     )

#     model_config = ConfigDict(extra="forbid")

#     @model_validator(mode="after")
#     def validate_cv_settings(self) -> "CVConfig":
#         if self.n_folds < 2:
#             raise ValueError("n_folds must be at least 2")
#         if self.strategy == "timeseries" and self.shuffle:
#             self.shuffle = False  # Timeseries should never shuffle
#             # You could also raise warning or error here
#         return self


# class TrainingConfig(BaseModel):
#     test_size: float = Field(0.2, gt=0.0, lt=1.0)
#     random_state: int = 42
#     batch_size: int = Field(64, ge=1)
#     max_epochs: int = Field(100, ge=1)
#     patience: int = Field(10, ge=1)
#     cv: CVConfig = Field(
#         default_factory=CVConfig,
#         description="Cross-validation settings for more robust HPO"
#     )

#     model_config = ConfigDict(extra="forbid")


# class OptunaConfig(BaseModel):
#     n_trials: int = Field(15, ge=1)
#     n_epochs_per_trial: int = Field(5, ge=1)
#     layer_range: Tuple[int, int] = Field((1, 4), description="min/max hidden layers")
#     units_list: List[int] = Field(default_factory=lambda: [16, 64, 128, 256])
#     dropout_range: Tuple[float, float] = Field((0.1, 0.4))
#     lr_range: Tuple[float, float] = Field((0.0001, 0.01))

#     sampler: Literal["tpe", "random", "grid"] = "tpe"          # add more if you use them
#     pruner: Literal["median", "hyperband", "none"] = "median"  # adjust as needed

#     model_config = ConfigDict(extra="forbid")

#     @field_validator("layer_range")
#     @classmethod
#     def check_layer_range(cls, v: Tuple[int, int]) -> Tuple[int, int]:
#         lo, hi = v
#         if lo >= hi or lo < 1:
#             raise ValueError(f"Invalid layer_range: {v} (lo < hi and lo >= 1)")
#         return v

#     @field_validator("dropout_range")
#     @classmethod
#     def check_dropout_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
#         lo, hi = v
#         if lo >= hi or lo < 0 or hi > 1:
#             raise ValueError(f"Invalid dropout_range: {v} (0 ≤ lo < hi ≤ 1)")
#         return v

#     @field_validator("lr_range")
#     @classmethod
#     def check_lr_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
#         lo, hi = v
#         if lo >= hi or lo <= 0:
#             raise ValueError(f"Invalid lr_range: {v} (0 < lo < hi)")
#         return v

# class ExportConfig(BaseModel):
#     dir: Path = Field(default_factory=lambda: Path("model_export"))
#     weights: str = "champion_weights.pth"
#     in_scaler: str = "input_scaler.pkl"
#     tar_scaler: str = "target_scaler.pkl"
#     cat_encoder: str = "categorical_input_encoders.pkl"
#     metadata: str = "metadata.json"

#     model_config = ConfigDict(extra="forbid")

#     @field_validator("dir", mode="before")
#     @classmethod
#     def resolve_export_dir(cls, v: str | Path) -> Path:
#         return Path(v).resolve()

# class RootConfig(BaseModel):
#     data: DataConfig
#     training: TrainingConfig
#     optuna: OptunaConfig
#     export: ExportConfig

#     project_root: Path | None = None
#     seed: int = 42
#     device: Literal["cuda", "cpu", "mps"] = "cuda"

#     model_config = ConfigDict(
#         extra="forbid",
#         validate_default=True,      # validate defaults too
#     )

#     @model_validator(mode="after")
#     def resolve_paths_relative_to_root(self) -> "RootConfig":
#         if self.project_root is None:
#             raise ValueError("project_root must be set")

#         root = self.project_root

#         # Make data path absolute if relative
#         if not self.data.train_data_path.is_absolute():
#             self.data.train_data_path = root / self.data.train_data_path

#         # Same for export dir
#         if not self.export.dir.is_absolute():
#             self.export.dir = root / self.export.dir

#         return self