# tests/test_inference_pipeline.py
import json
import joblib
import pytest
import pandas as pd
import torch
from pathlib import Path

from demo_ml_project.configs.inference.schema import InferenceConfig, LocalLoadConfig, InputConfig, OutputConfig
from demo_ml_project.pipelines.inference_pipeline import InferencePipeline, InferenceDataset
from demo_ml_project.models.model import DynamicTabularModel
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


@pytest.fixture
def sample_inference_data():
    """Create sample data for inference testing."""
    return pd.DataFrame({
        "category": ["A", "B", "A", "C"],
        "value": [10.0, 20.0, 15.0, 30.0],
    })


@pytest.fixture
def trained_model_artifacts(tmp_path):
    """Create minimal trained model artifacts for testing."""
    export_dir = tmp_path / "model_export"
    export_dir.mkdir()

    # Create model
    emb_sizes = [(3, 2)]  # 3 categories, embedding dim 2
    model = DynamicTabularModel(
        emb_sizes=emb_sizes,
        n_numeric=1,
        n_targets=1,
        hidden_dims=[8],
        dropout=0.0,
    )

    # Save model state
    torch.save(model.state_dict(), export_dir / "model_state.pth")

    # Create and save preprocessing artifacts
    cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_encoder.fit(pd.DataFrame({"category": ["A", "B", "C"]}))
    joblib.dump(cat_encoder, export_dir / "cat_encoder.joblib")

    num_scaler = StandardScaler()
    num_scaler.fit([[10.0], [20.0], [30.0]])
    joblib.dump(num_scaler, export_dir / "input_scaler.joblib")

    tar_scaler = StandardScaler()
    tar_scaler.fit([[1.0], [2.0], [3.0]])
    joblib.dump(tar_scaler, export_dir / "target_scaler.joblib")

    # Save metadata
    metadata = {
        "embedding_sizes": emb_sizes,
        "categorical_features": ["category"],
        "numeric_features": ["value"],
        "target_features": ["target"],
        "best_hparams": {
            "n_layers": 1,
            "n_units_l0": 8,
            "dropout": 0.0,
        },
    }

    with open(export_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return export_dir


@pytest.fixture
def inference_config(tmp_path, trained_model_artifacts):
    """Create inference configuration."""
    input_file = tmp_path / "input_data.csv"

    return InferenceConfig(
        load_from="local",
        device="cpu",
        batch_size=2,
        project_root=tmp_path,
        local=LocalLoadConfig(export_dir=trained_model_artifacts),
        input=InputConfig(path=input_file, format="csv"),
        output=OutputConfig(
            path=tmp_path / "predictions.csv",
            include_index=False,
            columns_prefix="pred_",
        ),
    )


class TestInferenceDataset:
    """Test suite for InferenceDataset."""

    def test_inference_dataset_basic(self, sample_inference_data):
        """Test basic InferenceDataset functionality."""
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_encoder.fit(sample_inference_data[["category"]])

        num_scaler = StandardScaler()
        num_scaler.fit(sample_inference_data[["value"]])

        dataset = InferenceDataset(
            df=sample_inference_data,
            cat_cols=["category"],
            num_cols=["value"],
            cat_encoder=cat_encoder,
            num_scaler=num_scaler,
        )

        assert len(dataset) == len(sample_inference_data)

        x_cat, x_num = dataset[0]
        assert x_cat.dtype == torch.long
        assert x_num.dtype == torch.float32

    def test_inference_dataset_missing_columns(self, sample_inference_data):
        """Test that dataset raises error for missing columns."""
        with pytest.raises(ValueError, match="missing required columns"):
            InferenceDataset(
                df=sample_inference_data,
                cat_cols=["category", "missing_col"],
                num_cols=["value"],
                cat_encoder=None,
                num_scaler=None,
            )

    def test_inference_dataset_no_categorical(self, sample_inference_data):
        """Test dataset with no categorical features."""
        num_scaler = StandardScaler()
        num_scaler.fit(sample_inference_data[["value"]])

        dataset = InferenceDataset(
            df=sample_inference_data,
            cat_cols=[],
            num_cols=["value"],
            cat_encoder=None,
            num_scaler=num_scaler,
        )

        x_cat, x_num = dataset[0]
        assert x_cat.shape == (0,)
        assert x_num.shape == (1,)

    def test_inference_dataset_no_numerical(self, sample_inference_data):
        """Test dataset with no numerical features."""
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_encoder.fit(sample_inference_data[["category"]])

        dataset = InferenceDataset(
            df=sample_inference_data,
            cat_cols=["category"],
            num_cols=[],
            cat_encoder=cat_encoder,
            num_scaler=None,
        )

        x_cat, x_num = dataset[0]
        assert x_cat.shape == (1,)
        assert x_num.shape == (0,)

    def test_inference_dataset_unknown_category(self, sample_inference_data):
        """Test handling of unknown categorical values."""
        # Train on subset of categories
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_encoder.fit(pd.DataFrame({"category": ["A", "B"]}))  # Only A and B

        # Test data includes C which is unknown
        dataset = InferenceDataset(
            df=sample_inference_data,
            cat_cols=["category"],
            num_cols=["value"],
            cat_encoder=cat_encoder,
            num_scaler=StandardScaler().fit(sample_inference_data[["value"]]),
        )

        # Should not raise error, unknown values get encoded as -1
        x_cat, x_num = dataset[3]  # Row with "C"
        assert x_cat[0] == -1  # Unknown value


class TestInferencePipeline:
    """Test suite for InferencePipeline."""

    def test_load_model_and_preprocessors_local(
        self, inference_config, trained_model_artifacts
    ):
        """Test loading model and preprocessors from local directory."""
        pipeline = InferencePipeline(inference_config)
        pipeline.load_model_and_preprocessors()

        assert pipeline.model is not None
        assert pipeline.cat_encoder is not None
        assert pipeline.num_scaler is not None
        assert pipeline.tar_scaler is not None
        assert pipeline.input_cat_cols == ["category"]
        assert pipeline.input_num_cols == ["value"]
        assert pipeline.target_features == ["target"]

    def test_load_model_missing_metadata(self, inference_config, trained_model_artifacts):
        """Test error handling when metadata.json is missing."""
        # Remove metadata file
        (trained_model_artifacts / "metadata.json").unlink()

        pipeline = InferencePipeline(inference_config)
        with pytest.raises(FileNotFoundError, match="metadata.json"):
            pipeline.load_model_and_preprocessors()

    def test_load_model_missing_weights(self, inference_config, trained_model_artifacts):
        """Test error handling when model weights are missing."""
        # Remove model weights
        (trained_model_artifacts / "model_state.pth").unlink()

        pipeline = InferencePipeline(inference_config)
        with pytest.raises(FileNotFoundError):
            pipeline.load_model_and_preprocessors()

    def test_predict_basic(
        self, tmp_path, sample_inference_data, inference_config, trained_model_artifacts
    ):
        """Test basic prediction functionality."""
        # Save input data
        input_path = tmp_path / "input_data.csv"
        sample_inference_data.to_csv(input_path, index=False)

        pipeline = InferencePipeline(inference_config)
        predictions = pipeline.predict()

        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == len(sample_inference_data)
        assert "pred_target" in predictions.columns

    def test_predict_missing_input_file(self, inference_config):
        """Test error when input file doesn't exist."""
        pipeline = InferencePipeline(inference_config)
        pipeline.load_model_and_preprocessors()

        with pytest.raises(FileNotFoundError, match="Input not found"):
            pipeline.predict()

    def test_predict_missing_required_columns(
        self, tmp_path, inference_config, trained_model_artifacts
    ):
        """Test error when input data is missing required columns."""
        # Create input with missing column
        bad_data = pd.DataFrame({"category": ["A", "B"]})  # Missing 'value'
        input_path = tmp_path / "input_data.csv"
        bad_data.to_csv(input_path, index=False)

        pipeline = InferencePipeline(inference_config)
        with pytest.raises(ValueError, match="missing.*required columns"):
            pipeline.predict()

    def test_predict_extra_columns(
        self, tmp_path, sample_inference_data, inference_config, trained_model_artifacts
    ):
        """Test that extra columns in input are handled gracefully."""
        # Add extra columns
        data_with_extra = sample_inference_data.copy()
        data_with_extra["extra1"] = [1, 2, 3, 4]
        data_with_extra["extra2"] = ["x", "y", "z", "w"]

        input_path = tmp_path / "input_data.csv"
        data_with_extra.to_csv(input_path, index=False)

        pipeline = InferencePipeline(inference_config)
        predictions = pipeline.predict()

        # Should work, extra columns ignored
        assert len(predictions) == len(data_with_extra)

    def test_predict_parquet_input(
        self, tmp_path, sample_inference_data, inference_config, trained_model_artifacts
    ):
        """Test prediction with Parquet input file."""
        # Update config for parquet
        input_path = tmp_path / "input_data.parquet"
        sample_inference_data.to_parquet(input_path, index=False)

        inference_config.input.path = input_path
        inference_config.input.format = "parquet"

        pipeline = InferencePipeline(inference_config)
        predictions = pipeline.predict()

        assert len(predictions) == len(sample_inference_data)

    def test_predict_auto_format_detection(
        self, tmp_path, sample_inference_data, inference_config, trained_model_artifacts
    ):
        """Test automatic format detection."""
        input_path = tmp_path / "input_data.parquet"
        sample_inference_data.to_parquet(input_path, index=False)

        inference_config.input.path = input_path
        inference_config.input.format = "auto"

        pipeline = InferencePipeline(inference_config)
        predictions = pipeline.predict()

        assert len(predictions) == len(sample_inference_data)

    def test_predict_batch_processing(
        self, tmp_path, inference_config, trained_model_artifacts
    ):
        """Test that batch size doesn't affect results."""
        # Create larger dataset
        large_data = pd.DataFrame({
            "category": ["A", "B", "C"] * 10,
            "value": list(range(30)),
        })

        input_path = tmp_path / "input_data.csv"
        large_data.to_csv(input_path, index=False)

        # Test with different batch sizes
        inference_config.batch_size = 2
        pipeline1 = InferencePipeline(inference_config)
        pred1 = pipeline1.predict()

        inference_config.batch_size = 10
        pipeline2 = InferencePipeline(inference_config)
        pred2 = pipeline2.predict()

        # Results should be identical regardless of batch size
        assert len(pred1) == len(pred2)
        pd.testing.assert_frame_equal(pred1, pred2)

    def test_predict_column_prefix(
        self, tmp_path, sample_inference_data, inference_config, trained_model_artifacts
    ):
        """Test output column prefix."""
        input_path = tmp_path / "input_data.csv"
        sample_inference_data.to_csv(input_path, index=False)

        inference_config.output.columns_prefix = "predicted_"

        pipeline = InferencePipeline(inference_config)
        predictions = pipeline.predict()

        assert "predicted_target" in predictions.columns

    def test_device_auto_detection(self, inference_config, trained_model_artifacts):
        """Test automatic device detection."""
        inference_config.device = None  # Auto-detect

        pipeline = InferencePipeline(inference_config)
        assert pipeline.device is not None
        assert pipeline.device.type in ["cpu", "cuda", "mps"]

    def test_device_explicit_cpu(self, inference_config, trained_model_artifacts):
        """Test explicit CPU device."""
        inference_config.device = "cpu"

        pipeline = InferencePipeline(inference_config)
        assert pipeline.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self, inference_config, trained_model_artifacts):
        """Test CUDA device when available."""
        inference_config.device = "cuda"

        pipeline = InferencePipeline(inference_config)
        pipeline.load_model_and_preprocessors()

        assert pipeline.device.type == "cuda"
        assert next(pipeline.model.parameters()).device.type == "cuda"

    def test_multiple_predictions(
        self, tmp_path, sample_inference_data, inference_config, trained_model_artifacts
    ):
        """Test running predictions multiple times."""
        input_path = tmp_path / "input_data.csv"
        sample_inference_data.to_csv(input_path, index=False)

        pipeline = InferencePipeline(inference_config)

        pred1 = pipeline.predict()
        pred2 = pipeline.predict()

        # Should be identical (model in eval mode, no randomness)
        pd.testing.assert_frame_equal(pred1, pred2)
