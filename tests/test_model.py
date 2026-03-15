# tests/test_model.py
import pytest
import torch

from demo_ml_project.models.model import DynamicTabularModel


class TestDynamicTabularModel:
    """Test suite for model architecture validation."""

    @pytest.fixture
    def basic_model_config(self):
        """Basic model configuration for testing."""
        return {
            "emb_sizes": [(10, 5), (20, 8)],  # 2 categorical features
            "n_numeric": 3,  # 3 numerical features
            "n_targets": 2,  # 2 target outputs
            "hidden_dims": [64, 32],
            "dropout": 0.1,
        }

    def test_model_instantiation(self, basic_model_config):
        """Test that model can be instantiated with valid config."""
        model = DynamicTabularModel(**basic_model_config)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_model_forward_pass_shape(self, basic_model_config):
        """Test that forward pass produces correct output shape."""
        model = DynamicTabularModel(**basic_model_config)
        model.eval()

        batch_size = 16
        n_cat = len(basic_model_config["emb_sizes"])
        n_num = basic_model_config["n_numeric"]
        n_targets = basic_model_config["n_targets"]

        # Create dummy input: concatenated [cat_features, num_features]
        # Model expects: first columns are categorical, last n_numeric columns are numerical
        total_features = n_cat + n_num
        x = torch.randn(batch_size, total_features)
        # Set categorical features to valid indices
        for i in range(n_cat):
            vocab_size = basic_model_config["emb_sizes"][i][0]
            x[:, i] = torch.randint(0, vocab_size, (batch_size,)).float()

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, n_targets)

    @pytest.mark.xfail(reason="Model has bug with empty embeddings when n_cat=0")
    def test_model_with_no_categorical_features(self):
        """Test model with only numerical features."""
        model = DynamicTabularModel(
            emb_sizes=[],  # No categorical features
            n_numeric=5,
            n_targets=1,
            hidden_dims=[32, 16],
            dropout=0.2,
        )

        batch_size = 8
        x = torch.randn(batch_size, 5)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 1)

    @pytest.mark.xfail(reason="Model has slicing bug when n_numeric=0")
    def test_model_with_no_numerical_features(self):
        """Test model with only categorical features."""
        model = DynamicTabularModel(
            emb_sizes=[(15, 7), (30, 10)],
            n_numeric=0,  # No numerical features
            n_targets=3,
            hidden_dims=[48],
            dropout=0.15,
        )

        batch_size = 10
        # Only categorical features
        x = torch.randint(0, 10, (batch_size, 2)).float()

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 3)

    def test_model_with_single_hidden_layer(self):
        """Test model with single hidden layer."""
        model = DynamicTabularModel(
            emb_sizes=[(5, 3)],
            n_numeric=2,
            n_targets=1,
            hidden_dims=[16],  # Single layer
            dropout=0.0,
        )

        batch_size = 4
        x = torch.randn(batch_size, 3)  # 1 cat + 2 num
        x[:, 0] = torch.randint(0, 5, (batch_size,)).float()  # Valid categorical indices

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 1)

    def test_model_trainable_parameters(self, basic_model_config):
        """Test that model has trainable parameters."""
        model = DynamicTabularModel(**basic_model_config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0
        assert trainable_params == total_params  # All params should be trainable

    def test_model_gradient_flow(self, basic_model_config):
        """Test that gradients flow through the model during backprop."""
        model = DynamicTabularModel(**basic_model_config)
        model.train()

        batch_size = 8
        n_cat = len(basic_model_config["emb_sizes"])
        n_num = basic_model_config["n_numeric"]
        n_features = n_cat + n_num
        n_targets = basic_model_config["n_targets"]

        x = torch.randn(batch_size, n_features)
        # Set valid categorical indices
        for i in range(n_cat):
            vocab_size = basic_model_config["emb_sizes"][i][0]
            x[:, i] = torch.randint(0, vocab_size, (batch_size,)).float()

        target = torch.randn(batch_size, n_targets)

        # Forward pass
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check that at least some gradients are non-zero
        has_gradients = any(
            p.grad is not None and torch.any(p.grad != 0)
            for p in model.parameters()
        )
        assert has_gradients, "Model should have non-zero gradients after backprop"

    def test_model_dropout_behavior(self, basic_model_config):
        """Test that dropout behaves differently in train vs eval mode."""
        model = DynamicTabularModel(**basic_model_config)

        batch_size = 100
        n_cat = len(basic_model_config["emb_sizes"])
        n_num = basic_model_config["n_numeric"]
        n_features = n_cat + n_num

        x = torch.randn(batch_size, n_features)
        # Set valid categorical indices
        for i in range(n_cat):
            vocab_size = basic_model_config["emb_sizes"][i][0]
            x[:, i] = torch.randint(0, vocab_size, (batch_size,)).float()

        # Get outputs in train mode (with dropout)
        model.train()
        with torch.no_grad():
            output_train_1 = model(x)
            output_train_2 = model(x)

        # Get outputs in eval mode (no dropout)
        model.eval()
        with torch.no_grad():
            output_eval_1 = model(x)
            output_eval_2 = model(x)

        # In eval mode, outputs should be identical
        assert torch.allclose(output_eval_1, output_eval_2, atol=1e-6)

        # In train mode, outputs might differ due to dropout (if dropout > 0)
        # This is probabilistic, but with dropout=0.1 and large batch, very likely to differ
        if basic_model_config["dropout"] > 0:
            # At least one output should differ
            assert not torch.allclose(output_train_1, output_train_2, atol=1e-6)

    def test_model_device_compatibility(self, basic_model_config):
        """Test that model can be moved to different devices."""
        model = DynamicTabularModel(**basic_model_config)

        # Test CPU
        model_cpu = model.to("cpu")
        assert next(model_cpu.parameters()).device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to("cuda")
            assert next(model_cuda.parameters()).device.type == "cuda"

    def test_model_state_dict_save_load(self, basic_model_config, tmp_path):
        """Test that model can be saved and loaded."""
        model = DynamicTabularModel(**basic_model_config)

        # Save state dict
        save_path = tmp_path / "model_state.pth"
        torch.save(model.state_dict(), save_path)

        # Create new model and load state dict
        model_loaded = DynamicTabularModel(**basic_model_config)
        model_loaded.load_state_dict(torch.load(save_path, weights_only=True))

        # Test that outputs are identical
        n_cat = len(basic_model_config["emb_sizes"])
        n_num = basic_model_config["n_numeric"]
        x = torch.randn(4, n_cat + n_num)
        # Set valid categorical indices
        for i in range(n_cat):
            vocab_size = basic_model_config["emb_sizes"][i][0]
            x[:, i] = torch.randint(0, vocab_size, (4,)).float()

        model.eval()
        model_loaded.eval()

        with torch.no_grad():
            output_original = model(x)
            output_loaded = model_loaded(x)

        assert torch.allclose(output_original, output_loaded, atol=1e-6)

    @pytest.mark.parametrize("n_targets", [1, 2, 5, 10])
    def test_model_multiple_targets(self, n_targets):
        """Test model with different numbers of target outputs."""
        model = DynamicTabularModel(
            emb_sizes=[(10, 5)],
            n_numeric=3,
            n_targets=n_targets,
            hidden_dims=[32],
            dropout=0.1,
        )

        batch_size = 8
        x = torch.randn(batch_size, 4)  # 1 cat + 3 num
        x[:, 0] = torch.randint(0, 10, (batch_size,)).float()  # Valid categorical indices

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, n_targets)

    def test_model_embedding_dimensions(self):
        """Test that embeddings have correct dimensions."""
        emb_sizes = [(10, 5), (20, 8), (30, 12)]
        model = DynamicTabularModel(
            emb_sizes=emb_sizes,
            n_numeric=2,
            n_targets=1,
            hidden_dims=[32],
            dropout=0.1,
        )

        # Check embedding layers
        assert len(model.embeddings) == len(emb_sizes)
        for i, (vocab_size, emb_dim) in enumerate(emb_sizes):
            assert model.embeddings[i].num_embeddings == vocab_size
            assert model.embeddings[i].embedding_dim == emb_dim
