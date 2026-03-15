# tests/test_preprocessing_artifacts.py
import pytest
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from demo_ml_project.data.processing import prepare_data


class TestPreprocessingArtifactPersistence:
    """Test suite for preprocessing artifact export and loading."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            "cat_a": ["red", "blue", "red", "green", "blue"],
            "cat_b": ["small", "large", "small", "medium", "large"],
            "num_1": [10.0, 20.0, 15.0, 30.0, 25.0],
            "num_2": [1.5, 2.5, 1.8, 3.2, 2.9],
            "target_1": [100.0, 200.0, 150.0, 300.0, 250.0],
            "target_2": [5.0, 10.0, 7.5, 15.0, 12.5],
        })

    @pytest.fixture
    def preprocessing_config(self, tmp_path, sample_data):
        """Create config for preprocessing."""
        from demo_ml_project.configs.training.schema import (
            RootConfig, GlobalConfig, DataConfig, TrainingConfig,
            OptunaConfig, ExportConfig, CVConfig
        )

        # Create a dummy data file to pass validation
        dummy_file = tmp_path / "dummy.parquet"
        sample_data.to_parquet(dummy_file)

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
                    cat_cols=["cat_a", "cat_b"],
                    date_cols=[],
                    num_cols=["num_1", "num_2"],
                    target_cols=["target_1", "target_2"],
                ),
                "training": TrainingConfig(
                    test_size=0.2,
                    random_state=42,
                    batch_size=2,
                    max_epochs=2,
                    patience=1,
                    cv=CVConfig(enabled=False),
                ),
                "optuna": OptunaConfig(
                    n_trials=1,
                    n_epochs_per_trial=1,
                    layer_range=(1, 2),
                    units_list=[8],
                    dropout_range=(0.0, 0.1),
                    lr_range=(1e-3, 1e-2),
                    sampler="tpe",
                    pruner="median",
                ),
                "export": ExportConfig(
                    dir=tmp_path / "export",
                    weights="weights.pth",
                    metadata="metadata.json",
                ),
            }
        )

    def test_save_and_load_cat_encoder(self, sample_data, preprocessing_config, tmp_path):
        """Test saving and loading categorical encoder."""
        # Fit preprocessing
        artifacts = prepare_data(sample_data, preprocessing_config, fit=True)

        # Save encoder
        save_path = tmp_path / "cat_encoder.joblib"
        joblib.dump(artifacts.cat_encoder, save_path)

        # Load encoder
        loaded_encoder = joblib.load(save_path)

        # Test that loaded encoder produces same results
        original_encoded = artifacts.cat_encoder.transform(sample_data[["cat_a", "cat_b"]])
        loaded_encoded = loaded_encoder.transform(sample_data[["cat_a", "cat_b"]])

        assert (original_encoded == loaded_encoded).all()

    def test_save_and_load_num_scaler(self, sample_data, preprocessing_config, tmp_path):
        """Test saving and loading numerical scaler."""
        artifacts = prepare_data(sample_data, preprocessing_config, fit=True)

        # Save scaler
        save_path = tmp_path / "num_scaler.joblib"
        joblib.dump(artifacts.num_scaler, save_path)

        # Load scaler
        loaded_scaler = joblib.load(save_path)

        # Test that loaded scaler produces same results
        original_scaled = artifacts.num_scaler.transform(sample_data[["num_1", "num_2"]])
        loaded_scaled = loaded_scaler.transform(sample_data[["num_1", "num_2"]])

        assert (original_scaled == loaded_scaled).all()

    def test_save_and_load_tar_scaler(self, sample_data, preprocessing_config, tmp_path):
        """Test saving and loading target scaler."""
        artifacts = prepare_data(sample_data, preprocessing_config, fit=True)

        # Save scaler
        save_path = tmp_path / "tar_scaler.joblib"
        joblib.dump(artifacts.tar_scaler, save_path)

        # Load scaler
        loaded_scaler = joblib.load(save_path)

        # Test that loaded scaler produces same results
        original_scaled = artifacts.tar_scaler.transform(sample_data[["target_1", "target_2"]])
        loaded_scaled = loaded_scaler.transform(sample_data[["target_1", "target_2"]])

        assert (original_scaled == loaded_scaled).all()

    def test_unknown_categories_after_load(self, sample_data, preprocessing_config, tmp_path):
        """Test that loaded encoder handles unknown categories correctly."""
        # Fit on subset of data
        train_data = sample_data.iloc[:3]
        artifacts = prepare_data(train_data, preprocessing_config, fit=True)

        # Save and load encoder
        save_path = tmp_path / "cat_encoder.joblib"
        joblib.dump(artifacts.cat_encoder, save_path)
        loaded_encoder = joblib.load(save_path)

        # Test data with new category
        test_data = pd.DataFrame({
            "cat_a": ["red", "yellow"],  # "yellow" is unknown
            "cat_b": ["small", "xlarge"],  # "xlarge" is unknown
        })

        # Should not raise error, unknown values get encoded as -1
        encoded = loaded_encoder.transform(test_data)
        assert encoded[1, 0] == -1  # "yellow" → -1
        assert encoded[1, 1] == -1  # "xlarge" → -1

    def test_inverse_transform_after_load(self, sample_data, preprocessing_config, tmp_path):
        """Test that loaded scaler can inverse transform."""
        artifacts = prepare_data(sample_data, preprocessing_config, fit=True)

        # Save and load target scaler
        save_path = tmp_path / "tar_scaler.joblib"
        joblib.dump(artifacts.tar_scaler, save_path)
        loaded_scaler = joblib.load(save_path)

        # Transform and inverse transform
        scaled = loaded_scaler.transform(sample_data[["target_1", "target_2"]])
        inversed = loaded_scaler.inverse_transform(scaled)

        # Should recover original values
        pd.testing.assert_frame_equal(
            pd.DataFrame(inversed, columns=["target_1", "target_2"]),
            sample_data[["target_1", "target_2"]],
            check_dtype=False,
            atol=1e-5,
        )

    def test_scaler_statistics_preserved(self, sample_data, preprocessing_config, tmp_path):
        """Test that scaler statistics are preserved after save/load."""
        artifacts = prepare_data(sample_data, preprocessing_config, fit=True)

        # Save statistics before saving
        original_mean = artifacts.num_scaler.mean_.copy()
        original_std = artifacts.num_scaler.scale_.copy()

        # Save and load
        save_path = tmp_path / "num_scaler.joblib"
        joblib.dump(artifacts.num_scaler, save_path)
        loaded_scaler = joblib.load(save_path)

        # Check statistics match
        assert (loaded_scaler.mean_ == original_mean).all()
        assert (loaded_scaler.scale_ == original_std).all()

    def test_encoder_categories_preserved(self, sample_data, preprocessing_config, tmp_path):
        """Test that encoder categories are preserved after save/load."""
        artifacts = prepare_data(sample_data, preprocessing_config, fit=True)

        # Save categories before saving
        original_categories = [cat.tolist() for cat in artifacts.cat_encoder.categories_]

        # Save and load
        save_path = tmp_path / "cat_encoder.joblib"
        joblib.dump(artifacts.cat_encoder, save_path)
        loaded_encoder = joblib.load(save_path)

        # Check categories match
        loaded_categories = [cat.tolist() for cat in loaded_encoder.categories_]
        assert loaded_categories == original_categories

    def test_all_artifacts_save_load_consistency(self, sample_data, preprocessing_config, tmp_path):
        """Test that all artifacts can be saved and loaded together."""
        # Fit preprocessing
        artifacts = prepare_data(sample_data, preprocessing_config, fit=True)

        # Save all artifacts
        export_dir = tmp_path / "artifacts"
        export_dir.mkdir()

        joblib.dump(artifacts.cat_encoder, export_dir / "cat_encoder.joblib")
        joblib.dump(artifacts.num_scaler, export_dir / "input_scaler.joblib")
        joblib.dump(artifacts.tar_scaler, export_dir / "target_scaler.joblib")

        # Load all artifacts
        loaded_cat_encoder = joblib.load(export_dir / "cat_encoder.joblib")
        loaded_num_scaler = joblib.load(export_dir / "input_scaler.joblib")
        loaded_tar_scaler = joblib.load(export_dir / "target_scaler.joblib")

        # Process new data with loaded artifacts
        new_data = pd.DataFrame({
            "cat_a": ["red", "blue"],
            "cat_b": ["small", "large"],
            "num_1": [12.0, 22.0],
            "num_2": [1.6, 2.6],
            "target_1": [120.0, 220.0],
            "target_2": [6.0, 11.0],
        })

        # Transform with original
        original_result = prepare_data(
            new_data,
            preprocessing_config,
            fit=False,
            cat_encoder=artifacts.cat_encoder,
            num_scaler=artifacts.num_scaler,
            tar_scaler=artifacts.tar_scaler,
        )

        # Transform with loaded
        loaded_result = prepare_data(
            new_data,
            preprocessing_config,
            fit=False,
            cat_encoder=loaded_cat_encoder,
            num_scaler=loaded_num_scaler,
            tar_scaler=loaded_tar_scaler,
        )

        # Results should be identical
        pd.testing.assert_frame_equal(
            original_result.processed_df,
            loaded_result.processed_df,
            check_dtype=False,
        )

    def test_artifacts_directory_structure(self, sample_data, preprocessing_config, tmp_path):
        """Test creating proper directory structure for artifacts."""
        export_dir = tmp_path / "model_export" / "v1"
        export_dir.mkdir(parents=True)

        artifacts = prepare_data(sample_data, preprocessing_config, fit=True)

        # Save with proper naming convention
        joblib.dump(artifacts.cat_encoder, export_dir / "cat_encoder.joblib")
        joblib.dump(artifacts.num_scaler, export_dir / "input_scaler.joblib")
        joblib.dump(artifacts.tar_scaler, export_dir / "target_scaler.joblib")

        # Verify files exist
        assert (export_dir / "cat_encoder.joblib").exists()
        assert (export_dir / "input_scaler.joblib").exists()
        assert (export_dir / "target_scaler.joblib").exists()

    def test_empty_categorical_features(self, preprocessing_config, tmp_path):
        """Test handling when there are no categorical features."""
        data_no_cat = pd.DataFrame({
            "num_1": [10.0, 20.0, 15.0],
            "num_2": [1.5, 2.5, 1.8],
            "target_1": [100.0, 200.0, 150.0],
            "target_2": [5.0, 10.0, 7.5],
        })

        preprocessing_config.data.cat_cols = []

        artifacts = prepare_data(data_no_cat, preprocessing_config, fit=True)

        # Save and load (cat_encoder might be None or empty)
        if artifacts.cat_encoder is not None:
            save_path = tmp_path / "cat_encoder.joblib"
            joblib.dump(artifacts.cat_encoder, save_path)
            loaded = joblib.load(save_path)
            assert loaded is not None

    @pytest.mark.skip(reason="sklearn StandardScaler requires at least one feature")
    def test_empty_numerical_features(self, preprocessing_config, tmp_path):
        """Test handling when there are no numerical features."""
        data_no_num = pd.DataFrame({
            "cat_a": ["red", "blue", "red"],
            "cat_b": ["small", "large", "small"],
            "target_1": [100.0, 200.0, 150.0],
            "target_2": [5.0, 10.0, 7.5],
        })

        preprocessing_config.data.num_cols = []

        artifacts = prepare_data(data_no_num, preprocessing_config, fit=True)

        # Save and load (num_scaler might be None or empty)
        if artifacts.num_scaler is not None:
            save_path = tmp_path / "num_scaler.joblib"
            joblib.dump(artifacts.num_scaler, save_path)
            loaded = joblib.load(save_path)
            assert loaded is not None
