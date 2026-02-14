
# demo_ml_project/configs/inference/schema.py

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class LocalLoadConfig(BaseModel):
    export_dir: Path = Field(
        "model_export/latest",
        description="Directory containing exported model & preprocessors"
    )


class MLflowLoadConfig(BaseModel):
    model_name: str = Field("TabularMultiTargetRegressor")
    version: str = Field("latest", description="Model version or 'latest'")


class InputConfig(BaseModel):
    path: Path = Field(..., description="Path to input data file")
    format: Literal["auto", "csv", "parquet"] = Field("auto")

    @field_validator("path", mode="after")
    @classmethod
    def resolve_path(cls, v: Path) -> Path:
        return v.resolve()


class OutputConfig(BaseModel):
    path: Path = Field("predictions.csv")
    include_index: bool = Field(True)
    columns_prefix: str = Field("pred_")

# ########## Feast ##########
class FeastConfig(BaseModel):
    repo_path: str = Field("feature_repo")
    entity_column: str = Field("sample_id")
    feature_service: str = Field("inference_features")


class InferenceConfig(BaseModel):
    """Top-level config for inference."""

    target_cols: list[str] = Field(
        default_factory=lambda: ["breast", "lung_and_bronchus", "melanoma_of_the_skin"],
        description="Target variable names (used for output column naming: pred_ + name)"
    )

    load_from: Literal["local", "mlflow"] = Field("local")

    num_workers: int = Field(default=0, ge=0, le=16,
        description="DataLoader workers (0 = no multiprocessing, safe for debugging)"
    )
    batch_size: int = Field(256, ge=1)
    device: Optional[Literal["cuda", "cpu", "mps"]] = Field(
        None, description="null = auto (cuda > mps > cpu)"
    )

    # Optional: add project_root if needed for relative path resolution
    project_root: Optional[Path] = None

    local: LocalLoadConfig = Field(default_factory=LocalLoadConfig)
    mlflow: MLflowLoadConfig = Field(default_factory=MLflowLoadConfig)

    input: InputConfig
    output: OutputConfig

    # ########## Feast ##########
    project_root: Optional[Path] = None
    feast: FeastConfig = Field(default_factory=FeastConfig)


    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def resolve_paths(self) -> "InferenceConfig":
        if self.project_root is None:
            raise ValueError("project_root should be injected (usually from global)")

        root = self.project_root

        # Make input/output paths absolute if relative
        if not self.input.path.is_absolute():
            self.input.path = root / self.input.path

        if not self.output.path.is_absolute():
            self.output.path = root / self.output.path

        # Local export dir
        if self.load_from == "local" and not self.local.export_dir.is_absolute():
            self.local.export_dir = root / self.local.export_dir

        # ########## Feast ##########
        if not Path(self.feast.repo_path).is_absolute():
            self.feast.repo_path = str(root / self.feast.repo_path)

        return self