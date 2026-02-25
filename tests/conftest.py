# tests/conftest.py
# tests/conftest.py
import pytest
import pandas as pd
import torch
from pathlib import Path

from demo_ml_project.configs.training.schema import (
    RootConfig,
    DataConfig,
    TrainingConfig,
    OptunaConfig,
    ExportConfig,
    CVConfig,
)


@pytest.fixture
def sample_cfg(tmp_path, sample_df):
    """Minimal but complete RootConfig for testing (satisfies all required fields)"""
    from demo_ml_project.configs.training.schema import GlobalConfig

    # Create dummy file to pass validation
    dummy_file = tmp_path / "dummy.parquet"
    sample_df.to_parquet(dummy_file)

    return RootConfig(
        **{
            "global": GlobalConfig(
                project_root=tmp_path,
                seed=42,
                device="cpu",
            ),
            "data": DataConfig(
                train_data_path=dummy_file,
                drop_columns=[],
                cat_cols=["category"],
                date_cols=[],
                num_cols=["value"],
                target_cols=["target"],
            ),
            "training": TrainingConfig(
                test_size=0.2,
                random_state=42,
                batch_size=32,
                max_epochs=10,
                patience=5,
                cv=CVConfig(enabled=False),
            ),
            "optuna": OptunaConfig(
                n_trials=5,
                n_epochs_per_trial=3,
                layer_range=(1, 3),
                units_list=[32, 64, 128],
                dropout_range=(0.1, 0.3),
                lr_range=(1e-4, 1e-2),
                sampler="tpe",
                pruner="median",
            ),
            "export": ExportConfig(
                dir=tmp_path / "dummy_export",
                weights="weights.pth",
                metadata="metadata.json",
            ),
        }
    )


@pytest.fixture
def sample_df():
    """Small dummy dataframe for testing"""
    return pd.DataFrame({
        "category": ["A", "B", "A", "C", "B", "A"],
        "value": [10.0, 20.0, 15.0, 30.0, 25.0, 18.0],
        "target": [1.2, 3.4, 2.1, 4.5, 3.8, 2.9],
        "extra": [99, 88, 77, 66, 55, 44],  # ignored column
    })


@pytest.fixture
def device():
    return torch.device("cpu")