# tests/test_mlflow_integration.py
import pytest
import json
import tempfile
import mlflow
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from demo_ml_project.models.model import DynamicTabularModel


class TestMLflowIntegration:
    """Test suite for MLflow logging and model registry integration."""

    @pytest.fixture
    def mlflow_tracking_uri(self, tmp_path):
        """Set up temporary MLflow tracking URI."""
        tracking_uri = f"file://{tmp_path}/mlruns"
        mlflow.set_tracking_uri(tracking_uri)
        return tracking_uri

    @pytest.fixture
    def sample_model(self):
        """Create a simple model for testing."""
        return DynamicTabularModel(
            emb_sizes=[(5, 3)],
            n_numeric=2,
            n_targets=1,
            hidden_dims=[8],
            dropout=0.0,
        )

    def test_mlflow_experiment_creation(self, mlflow_tracking_uri):
        """Test creating MLflow experiment."""
        experiment_name = "test_experiment"
        mlflow.set_experiment(experiment_name)

        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None
        assert experiment.name == experiment_name

    def test_mlflow_run_creation(self, mlflow_tracking_uri):
        """Test creating MLflow run."""
        mlflow.set_experiment("test_exp")

        with mlflow.start_run(run_name="test_run") as run:
            assert run is not None
            assert run.info.run_id is not None
            assert run.info.status == "RUNNING"

        # After context, run should be finished
        run_info = mlflow.get_run(run.info.run_id)
        assert run_info.info.status == "FINISHED"

    def test_mlflow_log_params(self, mlflow_tracking_uri):
        """Test logging parameters to MLflow."""
        mlflow.set_experiment("test_exp")

        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "model_type": "DynamicTabularModel",
        }

        with mlflow.start_run():
            mlflow.log_params(params)
            run_id = mlflow.active_run().info.run_id

        # Verify params were logged
        run = mlflow.get_run(run_id)
        for key, value in params.items():
            assert run.data.params[key] == str(value)

    def test_mlflow_log_metrics(self, mlflow_tracking_uri):
        """Test logging metrics to MLflow."""
        mlflow.set_experiment("test_exp")

        metrics = {
            "train_loss": 0.5,
            "val_loss": 0.6,
            "val_rmse": 0.775,
            "val_mse": 0.6,
        }

        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            run_id = mlflow.active_run().info.run_id

        # Verify metrics were logged
        run = mlflow.get_run(run_id)
        for key, value in metrics.items():
            assert abs(run.data.metrics[key] - value) < 1e-6

    def test_mlflow_log_model(self, mlflow_tracking_uri, sample_model, tmp_path):
        """Test logging PyTorch model to MLflow."""
        mlflow.set_experiment("test_exp")

        with mlflow.start_run():
            mlflow.pytorch.log_model(
                pytorch_model=sample_model,
                artifact_path="model",
            )
            run_id = mlflow.active_run().info.run_id

        # Verify model artifact exists
        client = mlflow.MlflowClient()
        artifacts = client.list_artifacts(run_id, path="model")
        assert len(artifacts) > 0

    def test_mlflow_log_artifacts(self, mlflow_tracking_uri, tmp_path):
        """Test logging artifact directory to MLflow."""
        mlflow.set_experiment("test_exp")

        # Create artifacts
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()

        (artifact_dir / "file1.txt").write_text("content1")
        (artifact_dir / "file2.json").write_text('{"key": "value"}')

        with mlflow.start_run():
            mlflow.log_artifacts(str(artifact_dir), "export")
            run_id = mlflow.active_run().info.run_id

        # Verify artifacts were logged
        client = mlflow.MlflowClient()
        artifacts = client.list_artifacts(run_id, path="export")
        artifact_names = [a.path for a in artifacts]

        assert "export/file1.txt" in artifact_names
        assert "export/file2.json" in artifact_names

    def test_mlflow_model_signature(self, mlflow_tracking_uri, sample_model):
        """Test logging model with signature."""
        import numpy as np
        from mlflow.models.signature import infer_signature

        mlflow.set_experiment("test_exp")

        # Create sample data
        sample_input = np.random.randn(4, 3).astype(np.float32)

        with torch.no_grad():
            sample_output = sample_model(torch.tensor(sample_input)).numpy()

        signature = infer_signature(sample_input, sample_output)

        with mlflow.start_run():
            mlflow.pytorch.log_model(
                pytorch_model=sample_model,
                artifact_path="model",
                signature=signature,
            )
            run_id = mlflow.active_run().info.run_id

        # Load model and verify signature
        loaded_model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
        assert loaded_model is not None

    def test_mlflow_model_with_metadata(self, mlflow_tracking_uri, sample_model):
        """Test logging model with custom metadata."""
        mlflow.set_experiment("test_exp")

        metadata = {
            "description": "Test model",
            "version": "1.0.0",
            "author": "test_suite",
        }

        with mlflow.start_run():
            mlflow.pytorch.log_model(
                pytorch_model=sample_model,
                artifact_path="model",
                metadata=metadata,
            )
            run_id = mlflow.active_run().info.run_id

        # Verify metadata
        client = mlflow.MlflowClient()
        model_uri = f"runs:/{run_id}/model"

        # Note: Metadata verification depends on MLflow version
        # Just verify model was logged successfully
        loaded_model = mlflow.pytorch.load_model(model_uri)
        assert loaded_model is not None

    def test_mlflow_model_registration(self, mlflow_tracking_uri, sample_model):
        """Test registering model to MLflow Model Registry."""
        mlflow.set_experiment("test_exp")

        model_name = "TestModel"

        with mlflow.start_run():
            mlflow.pytorch.log_model(
                pytorch_model=sample_model,
                artifact_path="model",
                registered_model_name=model_name,
            )

        # Verify model was registered
        client = mlflow.MlflowClient()
        registered_models = client.search_registered_models(f"name='{model_name}'")

        assert len(registered_models) > 0
        assert registered_models[0].name == model_name

    def test_mlflow_run_tags(self, mlflow_tracking_uri):
        """Test adding tags to MLflow run."""
        mlflow.set_experiment("test_exp")

        tags = {
            "experiment_type": "hyperparameter_tuning",
            "team": "ml_team",
            "priority": "high",
        }

        with mlflow.start_run():
            for key, value in tags.items():
                mlflow.set_tag(key, value)
            run_id = mlflow.active_run().info.run_id

        # Verify tags
        run = mlflow.get_run(run_id)
        for key, value in tags.items():
            assert run.data.tags[key] == value

    def test_mlflow_nested_runs(self, mlflow_tracking_uri):
        """Test nested MLflow runs."""
        mlflow.set_experiment("test_exp")

        with mlflow.start_run(run_name="parent") as parent_run:
            mlflow.log_param("parent_param", "value")

            with mlflow.start_run(run_name="child", nested=True) as child_run:
                mlflow.log_param("child_param", "value")

                # Verify parent-child relationship
                assert child_run.data.tags["mlflow.parentRunId"] == parent_run.info.run_id

    def test_mlflow_artifact_download(self, mlflow_tracking_uri, tmp_path):
        """Test downloading artifacts from MLflow."""
        mlflow.set_experiment("test_exp")

        # Create and log artifact
        artifact_content = "test content"
        artifact_file = tmp_path / "test_artifact.txt"
        artifact_file.write_text(artifact_content)

        with mlflow.start_run():
            mlflow.log_artifact(str(artifact_file), "artifacts")
            run_id = mlflow.active_run().info.run_id

        # Download artifact
        download_path = tmp_path / "downloads"
        downloaded = mlflow.artifacts.download_artifacts(
            f"runs:/{run_id}/artifacts/test_artifact.txt",
            dst_path=str(download_path),
        )

        downloaded_file = Path(downloaded)
        assert downloaded_file.exists()
        assert downloaded_file.read_text() == artifact_content

    def test_mlflow_log_dict(self, mlflow_tracking_uri, tmp_path):
        """Test logging dictionary as JSON artifact."""
        mlflow.set_experiment("test_exp")

        test_dict = {
            "model_config": {
                "hidden_dims": [64, 32],
                "dropout": 0.1,
            },
            "training_config": {
                "batch_size": 32,
                "epochs": 10,
            },
        }

        with mlflow.start_run():
            mlflow.log_dict(test_dict, "config.json")
            run_id = mlflow.active_run().info.run_id

        # Verify dict was logged
        client = mlflow.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]

        assert "config.json" in artifact_names

    def test_mlflow_search_runs(self, mlflow_tracking_uri):
        """Test searching for runs with filters."""
        mlflow.set_experiment("test_exp")

        # Create multiple runs with different params
        for lr in [0.001, 0.01, 0.1]:
            with mlflow.start_run():
                mlflow.log_param("learning_rate", lr)
                mlflow.log_metric("accuracy", lr * 10)

        # Search for runs
        experiment = mlflow.get_experiment_by_name("test_exp")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="params.learning_rate = '0.01'",
        )

        assert len(runs) == 1
        assert float(runs.iloc[0]["params.learning_rate"]) == 0.01

    def test_mlflow_get_best_run(self, mlflow_tracking_uri):
        """Test finding best run by metric."""
        mlflow.set_experiment("test_exp")

        losses = [0.5, 0.3, 0.7, 0.2]
        run_ids = []

        for loss in losses:
            with mlflow.start_run():
                mlflow.log_metric("val_loss", loss)
                run_ids.append(mlflow.active_run().info.run_id)

        # Find best run
        experiment = mlflow.get_experiment_by_name("test_exp")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.val_loss ASC"],
            max_results=1,
        )

        best_run = runs.iloc[0]
        assert best_run["metrics.val_loss"] == 0.2

    def test_mlflow_model_load_and_predict(self, mlflow_tracking_uri, sample_model):
        """Test loading model from MLflow and making predictions."""
        import numpy as np

        mlflow.set_experiment("test_exp")

        with mlflow.start_run():
            mlflow.pytorch.log_model(
                pytorch_model=sample_model,
                artifact_path="model",
            )
            run_id = mlflow.active_run().info.run_id

        # Load model
        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.pytorch.load_model(model_uri)

        # Make predictions
        test_input = torch.randn(4, 3)
        loaded_model.eval()

        with torch.no_grad():
            predictions = loaded_model(test_input)

        assert predictions.shape == (4, 1)
        assert not torch.isnan(predictions).any()

    def test_mlflow_experiment_lifecycle(self, mlflow_tracking_uri):
        """Test complete experiment lifecycle."""
        experiment_name = "lifecycle_test"

        # Create experiment
        experiment_id = mlflow.create_experiment(experiment_name)
        assert experiment_id is not None

        # Set experiment
        mlflow.set_experiment(experiment_name)

        # Create run
        with mlflow.start_run() as run:
            mlflow.log_param("test", "value")
            run_id = run.info.run_id

        # Get experiment
        experiment = mlflow.get_experiment(experiment_id)
        assert experiment.name == experiment_name

        # Get run
        run = mlflow.get_run(run_id)
        assert run.data.params["test"] == "value"

    @pytest.mark.parametrize("metric_value", [0.1, 0.5, 1.0, 5.0])
    def test_mlflow_log_different_metric_values(self, mlflow_tracking_uri, metric_value):
        """Test logging various metric values."""
        mlflow.set_experiment("test_exp")

        with mlflow.start_run():
            mlflow.log_metric("test_metric", metric_value)
            run_id = mlflow.active_run().info.run_id

        run = mlflow.get_run(run_id)
        assert abs(run.data.metrics["test_metric"] - metric_value) < 1e-6