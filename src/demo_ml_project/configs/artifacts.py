
# src/demo_ml_project/configs/artifacts.py

# src/demo_ml_project/configs/artifacts.py

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
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

    
# from __future__ import annotations  # if you want to avoid importing torch/optuna at top-level

# from typing import Any, Dict, List, Tuple

# import pandas as pd
# from pydantic import BaseModel, ConfigDict, Field
# from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# # Optional: if you want stricter typing (requires torch/optuna installed)
# # from torch.nn import Module as TorchModule
# # from optuna import Study as OptunaStudy

# class PreprocessingArtifacts(BaseModel):
#     processed_df: pd.DataFrame = Field(..., description="Preprocessed & cleaned DataFrame")
#     cat_encoder: OrdinalEncoder = Field(..., description="Fitted encoder for categorical features")
#     num_scaler: StandardScaler = Field(..., description="Fitted scaler for numerical features")
#     tar_scaler: StandardScaler = Field(..., description="Fitted scaler for target(s)")

#     model_config = ConfigDict(
#         arbitrary_types_allowed=True,  # ← required for pandas/sklearn objects
#         extra="forbid",                # prevent accidental extra fields
#         # frozen=True,                 # optional: make immutable after creation
#     )

#     # Optional: if you ever want to add light validation
#     # @model_validator(mode="after")
#     # def check_shapes_consistent(self) -> "PreprocessingArtifacts":
#     #     if len(self.processed_df) == 0:
#     #         raise ValueError("Processed DataFrame is empty")
#     #     return self


# class TrainingArtifacts(BaseModel):
#     model: Any = Field(..., description="Trained model (usually torch.nn.Module)")
#     study: Any = Field(..., description="Completed Optuna study")
#     # Practical — full validation of torch.nn.Module or optuna.Study internals is overkill
#     emb_sizes: List[Tuple[int, int]] = Field(
#         ..., description="Embedding sizes: (num_categories, embedding_dim) per cat feature"
#     )
#     preprocessors: Dict[str, Any] = Field(
#         default_factory=dict,
#         description="Additional fitted preprocessors (imputers, rare-label encoders, etc.)"
#     )

#     model_config = ConfigDict(
#         arbitrary_types_allowed=True,  # needed for torch/optuna/sklearn objects
#         extra="forbid",
#         # json_encoders={  # optional — if you ever .model_dump_json() these
#         #     pd.DataFrame: lambda df: df.to_dict(orient="records"),
#         #     OrdinalEncoder: lambda enc: {"categories_": enc.categories_.tolist()},
#         #     StandardScaler: lambda sc: {"mean_": sc.mean_.tolist(), "scale_": sc.scale_.tolist()},
#         # }
#     )

#     # Optional: stricter typing example (uncomment if you import the real types)
#     # model: TorchModule
#     # study: OptunaStudy
