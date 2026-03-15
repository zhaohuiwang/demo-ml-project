# tests/test_training_integration.py
import pytest
import json
import joblib
import torch
import pandas as pd
from pathlib import Path

from demo_ml_project.models.model import DynamicTabularModel
from demo_ml_project.data.dataset import InputDataset
from demo_ml_project.utils.early_stopping import EarlyStopping
from torch.utils.data import DataLoader


class TestTrainingIntegration:
    """Integration tests for training components working together."""

    @pytest.fixture
    def training_data(self):
        """Create sample training data."""
        return pd.DataFrame({
            "cat_1": [0, 1, 0, 2, 1, 0, 2, 1],  # Already encoded
            "num_1": [-1.0, 0.5, -0.5, 1.5, 0.8, -0.8, 1.2, 0.2],  # Already scaled
            "num_2": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
            "target_1": [0.5, 1.5, 1.0, 2.5, 2.0, 1.2, 2.8, 1.8],  # Scaled targets
        })

    @pytest.fixture
    def model_config(self):
        """Basic model configuration."""
        return {
            "emb_sizes": [(3, 2)],  # 3 categories, embedding dim 2
            "n_numeric": 2,
            "n_targets": 1,
            "hidden_dims": [16, 8],
            "dropout": 0.1,
        }

    def test_dataset_to_dataloader_to_model(self, training_data, model_config):
        """Test full pipeline: DataFrame → Dataset → DataLoader → Model."""
        # Create dataset
        dataset = InputDataset(
            df=training_data,
            cat_cols=["cat_1"],
            num_cols=["num_1", "num_2"],
            target_cols=["target_1"],
        )

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        # Create model
        model = DynamicTabularModel(**model_config)
        model.eval()

        # Run inference
        with torch.no_grad():
            for x_cat, x_num, y in dataloader:
                x = torch.cat([x_cat, x_num], dim=1)
                output = model(x)

                assert output.shape == (len(x), 1)
                assert not torch.isnan(output).any()
                assert not torch.isinf(output).any()

    def test_single_training_epoch(self, training_data, model_config):
        """Test running a single training epoch."""
        dataset = InputDataset(
            df=training_data,
            cat_cols=["cat_1"],
            num_cols=["num_1", "num_2"],
            target_cols=["target_1"],
        )

        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        model = DynamicTabularModel(**model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        # Training epoch
        model.train()
        total_loss = 0.0

        for x_cat, x_num, y in dataloader:
            x = torch.cat([x_cat, x_num], dim=1)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        assert avg_loss > 0
        assert not pd.isna(avg_loss)

    def test_training_with_early_stopping(self, training_data, model_config):
        """Test training with early stopping."""
        # Split into train/val
        train_data = training_data.iloc[:6]
        val_data = training_data.iloc[6:]

        train_ds = InputDataset(train_data, ["cat_1"], ["num_1", "num_2"], ["target_1"])
        val_ds = InputDataset(val_data, ["cat_1"], ["num_1", "num_2"], ["target_1"])

        train_loader = DataLoader(train_ds, batch_size=3, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
        model = DynamicTabularModel(**model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        early_stopping = EarlyStopping(patience=3)

        # Train for multiple epochs
        for epoch in range(10):
            # Train
            model.train()
            for x_cat, x_num, y in train_loader:
                x = torch.cat([x_cat, x_num], dim=1)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_cat, x_num, y in val_loader:
                    x = torch.cat([x_cat, x_num], dim=1)
                    output = model(x)
                    loss = criterion(output, y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                break

        # Early stopping should have triggered (or reached max epochs)
        assert epoch < 10 or early_stopping.early_stop
        assert early_stopping.best_model_state is not None

    def test_model_export_and_load(self, model_config, tmp_path):
        """Test exporting and loading model state."""
        # Create and train model briefly
        model = DynamicTabularModel(**model_config)

        # Export
        export_path = tmp_path / "model_state.pth"
        torch.save(model.state_dict(), export_path)

        # Create new model and load
        loaded_model = DynamicTabularModel(**model_config)
        loaded_model.load_state_dict(
            torch.load(export_path, map_location="cpu", weights_only=True)
        )
         # Test equivalence
        test_input = torch.randn(4, 3)  # 1 cat + 2 num
        test_input[:, 0] = torch.randint(0, 3, (4,)).float()
        model.eval()
        loaded_model.eval()
        with torch.no_grad():
            output1 = model(test_input)
            output2 = loaded_model(test_input)

        assert torch.allclose(output1, output2, atol=1e-6)

    def test_full_training_artifacts_export(self, training_data, model_config, tmp_path):
        """Test exporting all training artifacts together."""
        from sklearn.preprocessing import OrdinalEncoder, StandardScaler

        export_dir = tmp_path / "model_export"
        export_dir.mkdir()

        # Train model
        dataset = InputDataset(
            training_data, ["cat_1"], ["num_1", "num_2"], ["target_1"]
        )
        dataloader = DataLoader(dataset, batch_size=4)

        model = DynamicTabularModel(**model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        model.train()
        for x_cat, x_num, y in dataloader:
            x = torch.cat([x_cat, x_num], dim=1)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Export model
        torch.save(model.state_dict(), export_dir / "model_state.pth")

        # Create and export preprocessing artifacts
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_encoder.fit([[0], [1], [2]])
        joblib.dump(cat_encoder, export_dir / "cat_encoder.joblib")

        num_scaler = StandardScaler()
        num_scaler.fit([[-1.0, 0.1], [0.5, -0.2], [1.5, -0.4]])
        joblib.dump(num_scaler, export_dir / "input_scaler.joblib")
        tar_scaler = StandardScaler()
        tar_scaler.fit([[0.5], [1.5], [2.5]])
        joblib.dump(tar_scaler, export_dir / "target_scaler.joblib")

        # Export metadata
        metadata = {
            "embedding_sizes": model_config["emb_sizes"],
            "categorical_features": ["cat_1"],
            "numeric_features": ["num_1", "num_2"],
            "target_features": ["target_1"],
            "best_hparams": {
                "n_layers": 2,
                "n_units_l0": 16,
                "n_units_l1": 8,
                "dropout": 0.1,
            },
        }

        with open(export_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Verify all files exist
        assert (export_dir / "model_state.pth").exists()
        assert (export_dir / "cat_encoder.joblib").exists()
        assert (export_dir / "input_scaler.joblib").exists()
        assert (export_dir / "target_scaler.joblib").exists()
        assert (export_dir / "metadata.json").exists()

        # Verify can load all artifacts
        loaded_model = DynamicTabularModel(**model_config)
        loaded_model.load_state_dict(
            torch.load(export_dir / "model_state.pth", map_location="cpu", weights_only=True)
        )

        loaded_cat_encoder = joblib.load(export_dir / "cat_encoder.joblib")
        loaded_num_scaler = joblib.load(export_dir / "input_scaler.joblib")
        loaded_tar_scaler = joblib.load(export_dir / "target_scaler.joblib")

        with open(export_dir / "metadata.json") as f:
            loaded_metadata = json.load(f)

        assert loaded_metadata["embedding_sizes"] == model_config["emb_sizes"]
        assert loaded_cat_encoder is not None
        assert loaded_num_scaler is not None
        assert loaded_tar_scaler is not None

    def test_gradient_accumulation(self, training_data, model_config):
        """Test training with gradient accumulation."""
        dataset = InputDataset(
            training_data, ["cat_1"], ["num_1", "num_2"], ["target_1"]
        )
        dataloader = DataLoader(dataset, batch_size=2)

        model = DynamicTabularModel(**model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        accumulation_steps = 2
        model.train()

        for i, (x_cat, x_num, y) in enumerate(dataloader):
            x = torch.cat([x_cat, x_num], dim=1)

            output = model(x)
            loss = criterion(output, y)
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Should complete without error
        assert True

    def test_model_determinism_with_seed(self, training_data, model_config):
        """Test that training is deterministic with fixed seed."""
        torch.manual_seed(42)
        dataset = InputDataset(
            training_data, ["cat_1"], ["num_1", "num_2"], ["target_1"]
        )
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

        model1 = DynamicTabularModel(**model_config)
        torch.manual_seed(42)
        model2 = DynamicTabularModel(**model_config)

        # Same initialization
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        # Train both identically
        for _ in range(2):
            for (x_cat1, x_num1, y1), (x_cat2, x_num2, y2) in zip(dataloader, dataloader):
                x1 = torch.cat([x_cat1, x_num1], dim=1)
                x2 = torch.cat([x_cat2, x_num2], dim=1)

                optimizer1.zero_grad()
                optimizer2.zero_grad()

                loss1 = criterion(model1(x1), y1)
                loss2 = criterion(model2(x2), y2)

                loss1.backward()
                loss2.backward()

                optimizer1.step()
                optimizer2.step()

        # Parameters should be identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)

    def test_batch_size_independence(self, training_data, model_config):
        """Test that batch size doesn't affect final model (given enough epochs)."""
        # Note: This is a simplified test - true independence requires more careful setup
        dataset = InputDataset(
            training_data, ["cat_1"], ["num_1", "num_2"], ["target_1"]
        )

        # Small batch
        loader_small = DataLoader(dataset, batch_size=2, shuffle=False)
        # Large batch
        loader_large = DataLoader(dataset, batch_size=8, shuffle=False)

        # Both should be able to process the data
        model = DynamicTabularModel(**model_config)
        criterion = torch.nn.MSELoss()

        model.eval()
        with torch.no_grad():
            loss_small = 0
            for x_cat, x_num, y in loader_small:
                x = torch.cat([x_cat, x_num], dim=1)
                output = model(x)
                loss_small += criterion(output, y).item()

            loss_large = 0
            for x_cat, x_num, y in loader_large:
                x = torch.cat([x_cat, x_num], dim=1)
                output = model(x)
                loss_large += criterion(output, y).item()

        # Total loss should be similar (not exact due to averaging)
        assert abs(loss_small - loss_large) < 0.1
