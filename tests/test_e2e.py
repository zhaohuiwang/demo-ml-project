# tests/test_e2e.py
"""End-to-end workflow tests."""
import pytest
import pandas as pd
import torch
import json
import joblib
from pathlib import Path

from demo_ml_project.models.model import DynamicTabularModel
from demo_ml_project.data.dataset import InputDataset
from demo_ml_project.data.processing import prepare_data
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


@pytest.fixture
def e2e_data():
    """Sample data for end-to-end testing."""
    return pd.DataFrame({
        "cat_1": ["A", "B", "A", "C", "B", "A", "C", "B"] * 3,
        "num_1": [10.0, 20.0, 15.0, 30.0, 25.0, 18.0, 28.0, 22.0] * 3,
        "num_2": [1.5, 2.5, 1.8, 3.2, 2.9, 2.1, 3.0, 2.7] * 3,
        "target": [100.0, 200.0, 150.0, 300.0, 250.0, 180.0, 280.0, 220.0] * 3,
    })


@pytest.fixture
def e2e_config(tmp_path):
    """Configuration for e2e testing."""
    from demo_ml_project.configs.training.schema import (
        RootConfig, DataConfig, TrainingConfig, OptunaConfig, ExportConfig, CVConfig
    )

    return RootConfig(
        project_root=tmp_path,
        seed=42,
        device="cpu",
        data=DataConfig(
            train_data_path=tmp_path / "data.parquet",
            drop_columns=[],
            cat_cols=["cat_1"],
            date_cols=[],
            num_cols=["num_1", "num_2"],
            target_cols=["target"],
        ),
        training=TrainingConfig(
            test_size=0.2,
            random_state=42,
            batch_size=4,
            max_epochs=3,
            patience=2,
            cv=CVConfig(enabled=False),
        ),
        optuna=OptunaConfig(
            n_trials=2,
            n_epochs_per_trial=2,
            layer_range=(1, 2),
            units_list=[8, 16],
            dropout_range=(0.0, 0.1),
            lr_range=(1e-3, 1e-2),
            sampler="tpe",
            pruner="median",
        ),
        export=ExportConfig(
            dir=tmp_path / "export",
            weights="model_state.pth",
            metadata="metadata.json",
        ),
    )


def test_full_workflow_preprocess_train_export(e2e_data, e2e_config, tmp_path):
    """Test complete workflow: preprocess → train → export."""

    # Step 1: Preprocess data
    artifacts = prepare_data(e2e_data, e2e_config, fit=True)
    processed_df = artifacts.processed_df

    assert len(processed_df) == len(e2e_data)
    assert artifacts.cat_encoder is not None
    assert artifacts.num_scaler is not None
    assert artifacts.tar_scaler is not None

    # Step 2: Create dataset and train model
    train_ds = InputDataset(
        processed_df,
        cat_cols=["cat_1"],
        num_cols=["num_1", "num_2"],
        target_cols=["target"],
    )

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    # Create model
    emb_sizes = [(int(e2e_data["cat_1"].nunique()), 2)]
    model = DynamicTabularModel(
        emb_sizes=emb_sizes,
        n_numeric=2,
        n_targets=1,
        hidden_dims=[16, 8],
        dropout=0.1,
    )

    # Train briefly
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(3):
        for x_cat, x_num, y in train_loader:
            x = torch.cat([x_cat, x_num], dim=1)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    # Step 3: Export everything
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    torch.save(model.state_dict(), export_dir / "model_state.pth")
    joblib.dump(artifacts.cat_encoder, export_dir / "cat_encoder.joblib")
    joblib.dump(artifacts.num_scaler, export_dir / "input_scaler.joblib")
    joblib.dump(artifacts.tar_scaler, export_dir / "target_scaler.joblib")

    metadata = {
        "embedding_sizes": emb_sizes,
        "categorical_features": ["cat_1"],
        "numeric_features": ["num_1", "num_2"],
        "target_features": ["target"],
        "best_hparams": {"n_layers": 2, "n_units_l0": 16, "n_units_l1": 8, "dropout": 0.1},
    }

    with open(export_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Step 4: Verify all files exist
    assert (export_dir / "model_state.pth").exists()
    assert (export_dir / "cat_encoder.joblib").exists()
    assert (export_dir / "input_scaler.joblib").exists()
    assert (export_dir / "target_scaler.joblib").exists()
    assert (export_dir / "metadata.json").exists()


def test_full_workflow_load_and_inference(e2e_data, e2e_config, tmp_path):
    """Test loading exported artifacts and running inference."""

    # Setup: train and export model
    artifacts = prepare_data(e2e_data, e2e_config, fit=True)
    processed_df = artifacts.processed_df

    train_ds = InputDataset(processed_df, ["cat_1"], ["num_1", "num_2"], ["target"])
    train_loader = DataLoader(train_ds, batch_size=4)

    emb_sizes = [(int(e2e_data["cat_1"].nunique()), 2)]
    model = DynamicTabularModel(emb_sizes=emb_sizes, n_numeric=2, n_targets=1, hidden_dims=[8], dropout=0.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    model.train()
    for x_cat, x_num, y in train_loader:
        x = torch.cat([x_cat, x_num], dim=1)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    # Export
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    torch.save(model.state_dict(), export_dir / "model_state.pth")
    joblib.dump(artifacts.cat_encoder, export_dir / "cat_encoder.joblib")
    joblib.dump(artifacts.num_scaler, export_dir / "input_scaler.joblib")
    joblib.dump(artifacts.tar_scaler, export_dir / "target_scaler.joblib")

    metadata = {
        "embedding_sizes": emb_sizes,
        "categorical_features": ["cat_1"],
        "numeric_features": ["num_1", "num_2"],
        "target_features": ["target"],
        "best_hparams": {"n_layers": 1, "n_units_l0": 8, "dropout": 0.0},
    }

    with open(export_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Load everything back
    with open(export_dir / "metadata.json") as f:
        loaded_meta = json.load(f)

    loaded_model = DynamicTabularModel(
        emb_sizes=loaded_meta["embedding_sizes"],
        n_numeric=len(loaded_meta["numeric_features"]),
        n_targets=len(loaded_meta["target_features"]),
        hidden_dims=[8],
        dropout=0.0,
    )

    loaded_model.load_state_dict(torch.load(export_dir / "model_state.pth", weights_only=True))
    loaded_cat_encoder = joblib.load(export_dir / "cat_encoder.joblib")
    loaded_num_scaler = joblib.load(export_dir / "input_scaler.joblib")
    loaded_tar_scaler = joblib.load(export_dir / "target_scaler.joblib")

    # Run inference on new data
    new_data = pd.DataFrame({
        "cat_1": ["A", "B"],
        "num_1": [12.0, 22.0],
        "num_2": [1.7, 2.7],
    })

    # Preprocess with loaded artifacts
    new_artifacts = prepare_data(
        new_data,
        e2e_config,
        fit=False,
        cat_encoder=loaded_cat_encoder,
        num_scaler=loaded_num_scaler,
        tar_scaler=loaded_tar_scaler,
    )

    # Inference
    inference_ds = InputDataset(new_artifacts.processed_df, ["cat_1"], ["num_1", "num_2"], ["target"])

    loaded_model.eval()
    with torch.no_grad():
        x_cat, x_num, _ = inference_ds[0]
        x = torch.cat([x_cat.unsqueeze(0), x_num.unsqueeze(0)], dim=1)
        pred_scaled = loaded_model(x)
        pred = loaded_tar_scaler.inverse_transform(pred_scaled.numpy())

    assert pred.shape == (1, 1)
    assert not pd.isna(pred[0, 0])


def test_preprocessing_consistency_across_splits(e2e_data, e2e_config):
    """Test that preprocessing is consistent when applied to different data splits."""

    # Split data
    train_data = e2e_data.iloc[:16]
    test_data = e2e_data.iloc[16:]

    # Fit on train
    train_artifacts = prepare_data(train_data, e2e_config, fit=True)

    # Transform test using train artifacts
    test_artifacts = prepare_data(
        test_data,
        e2e_config,
        fit=False,
        cat_encoder=train_artifacts.cat_encoder,
        num_scaler=train_artifacts.num_scaler,
        tar_scaler=train_artifacts.tar_scaler,
    )

    # Verify no data leakage - test stats should differ from train
    train_processed = train_artifacts.processed_df
    test_processed = test_artifacts.processed_df

    assert len(train_processed) == len(train_data)
    assert len(test_processed) == len(test_data)

    # Both should be properly scaled
    assert abs(train_processed["num_1"].mean()) < 0.5  # Approximately mean-centered
    assert abs(test_processed["num_1"].mean()) < 2.0