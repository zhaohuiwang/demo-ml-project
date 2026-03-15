# tests/test_mlflow_logging.py

import pytest
import mlflow
import torch
from pathlib import Path

from demo_ml_project.models.model import DynamicTabularModel


@pytest.fixture
def mlflow_tracking_uri(tmp_path):
    """Set up temporary MLflow tracking URI."""
    tracking_uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri
    mlflow.end_run()


@pytest.fixture
def sample_model():
    """Create a simple model for testing."""
    return DynamicTabularModel(
        emb_sizes=[(5, 3)],
        n_numeric=2,
        n_targets=1,
        hidden_dims=[8],
        dropout=0.0,
    )


def test_mlflow_log_params(mlflow_tracking_uri):
    """Test logging parameters to MLflow."""
    mlflow.set_experiment("test_exp")

    params = {"learning_rate": 0.001, "batch_size": 32}

    with mlflow.start_run():
        mlflow.log_params(params)
        run_id = mlflow.active_run().info.run_id

    run = mlflow.get_run(run_id)
    assert run.data.params["learning_rate"] == "0.001"
    assert run.data.params["batch_size"] == "32"


def test_mlflow_log_metrics(mlflow_tracking_uri):
    """Test logging metrics to MLflow."""
    mlflow.set_experiment("test_exp")

    with mlflow.start_run():
        mlflow.log_metrics({"val_loss": 0.5, "val_rmse": 0.707})
        run_id = mlflow.active_run().info.run_id

    run = mlflow.get_run(run_id)
    assert abs(run.data.metrics["val_loss"] - 0.5) < 1e-6


def test_mlflow_log_pytorch_model(mlflow_tracking_uri, sample_model):
    """Test logging PyTorch model to MLflow."""
    mlflow.set_experiment("test_exp")

    with mlflow.start_run():
        mlflow.pytorch.log_model(sample_model, "model")
        run_id = mlflow.active_run().info.run_id

    # Verify model can be loaded
    loaded = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    assert loaded is not None


def test_mlflow_log_artifacts(mlflow_tracking_uri, tmp_path):
    """Test logging artifacts directory."""
    mlflow.set_experiment("test_exp")

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "test.txt").write_text("content")

    with mlflow.start_run():
        mlflow.log_artifacts(str(artifact_dir), "export")
        run_id = mlflow.active_run().info.run_id

    client = mlflow.MlflowClient()
    artifacts = client.list_artifacts(run_id, "export")
    assert len(artifacts) > 0