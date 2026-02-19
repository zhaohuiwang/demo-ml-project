
# project-root(demo-ml-project)/src/demo_ml_project/models/model.py

import torch
import torch.nn as nn


class DynamicTabularModel(nn.Module):
    """
    Dynamic feed-forward neural network for tabular data with mixed features.

    - Categorical features are processed via learnable embeddings
    - Numerical features are concatenated directly
    - Multiple hidden layers with BatchNorm + ReLU + Dropout
    - Supports multi-target regression (or classification with appropriate head/loss)
    """

    def __init__(
        self,
        emb_sizes: list[tuple[int, int]], # (num_categories, embedding_dim) per cat feature
        n_numeric: int,
        n_targets: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        emb_sizes : list[tuple[int, int]]
            Embedding configuration for each categorical feature
        n_numeric : int
            Number of numerical input features
        n_targets : int
            Number of output targets (multi-target support)
        hidden_dims : list[int]
            Sizes of hidden layers in the MLP
        dropout : float
            Dropout probability after each hidden layer
        """
        super().__init__()
        self.n_numeric = n_numeric

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_cats, emb_dim) for num_cats, emb_dim in emb_sizes]
        )

        # Total embedding output dimension
        total_emb_dim = sum(emb_dim for _, emb_dim in emb_sizes)

        # MLP input dimension = embeddings + raw numerical features
        in_features = total_emb_dim + n_numeric

        # Build hidden layers
        layers: list[nn.Module] = []
        current_dim = in_features

        for hidden_size in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_size

        self.mlp = nn.Sequential(*layers) # Multilayer Perceptron

        # Final output layer
        self.output_head = nn.Linear(current_dim, n_targets)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            shape: (batch_size, Categorical + Numerical)

        Returns
        -------
        torch.Tensor
            Model predictions (batch_size, n_targets)
        """
        # Split categorical and numerical parts
        x_cat = x[:, : -self.n_numeric].long()
        x_num = x[:, -self.n_numeric :]
        
        # Embed categorical features
        embedded = [
            embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)
        ]
        x_emb = torch.cat(embedded, dim=1)

        # Concatenate embeddings + numerical features
        x_all = torch.cat([x_emb, x_num], dim=1)

        # Pass through MLP
        x_all = self.mlp(x_all)

        # Final prediction
        return self.output_head(x_all)