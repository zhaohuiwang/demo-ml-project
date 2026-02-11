
# src/demo_ml_project/configs/artifacts.py

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class PreprocessingArtifacts(BaseModel):
    """
    Container for fitted preprocessors and the resulting processed DataFrame.
    Used to ensure consistent transformation across train/validation/test sets.
    """

    processed_df: pd.DataFrame = Field(..., description="DataFrame after encoding and scaling")
    cat_encoder: OrdinalEncoder = Field(..., description="Fitted categorical ordinal encoder")
    num_scaler: StandardScaler = Field(..., description="Fitted scaler for numerical features")
    tar_scaler: StandardScaler = Field(..., description="Fitted scaler for target variables")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Required for pandas & sklearn objects
        extra="forbid",
        validate_assignment=True,
    )

    # Optional: if you ever want to add light validation
    @model_validator(mode="after")
    def check_shapes_consistent(self) -> "PreprocessingArtifacts":
        if len(self.processed_df) == 0:
            raise ValueError("Processed DataFrame is empty")
        return self


class TrainingArtifacts(BaseModel):
    """
    Final artifacts from the full training pipeline.
    Contains the trained model, Optuna study results, and all fitted preprocessors.
    """

    model: Any = Field(..., description="Fully trained final model (torch.nn.Module)")
    study: Any = Field(..., description="Completed Optuna study with best hyperparameters")
    emb_sizes: List[Tuple[int, int]] = Field(
        ..., description="Embedding sizes: (vocab_size, embedding_dim) per categorical feature"
    )
    preprocessors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Fitted preprocessors (cat_encoder, num_scaler, tar_scaler)"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # torch.nn.Module, optuna.Study, sklearn objects
        extra="forbid",
    )