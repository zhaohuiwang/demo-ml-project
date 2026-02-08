
# src/demo_ml_project/configs/artifacts.py

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
# If forward refs are annoying, you can import torch/optuna here instead

@dataclass
class PreprocessingArtifacts:
    processed_df: pd.DataFrame
    cat_encoder: OrdinalEncoder
    num_scaler: StandardScaler
    tar_scaler: StandardScaler

@dataclass
class TrainingArtifacts:
    model: Any          # or 'torch.nn.Module' with from __future__ import annotations
    study: Any          # or 'optuna.Study'
    emb_sizes: list[tuple[int, int]]
    preprocessors: dict[str, Any]