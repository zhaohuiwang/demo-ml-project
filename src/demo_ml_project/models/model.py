
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
        emb_sizes: list[tuple[int, int]],           # (num_categories, embedding_dim) per cat feature
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

        self.mlp = nn.Sequential(*layers)

        # Final output layer
        self.output_head = nn.Linear(current_dim, n_targets)

    def forward(
        self,
        x_cat: torch.Tensor,    # shape: (batch_size, num_cat_features)
        x_num: torch.Tensor,    # shape: (batch_size, n_numeric)
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x_cat : torch.Tensor
            Categorical feature indices
        x_num : torch.Tensor
            Numerical features (already scaled)

        Returns
        -------
        torch.Tensor
            Model predictions (batch_size, n_targets)
        """
        # Embed categorical features
        embedded = [
            embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)
        ]
        x_emb = torch.cat(embedded, dim=1)

        # Concatenate embeddings + numerical features
        x = torch.cat([x_emb, x_num], dim=1)

        # Pass through MLP
        x = self.mlp(x)

        # Final prediction
        return self.output_head(x)
    
# import torch
# import torch.nn as nn

# class DynamicModel(nn.Module):
#     """ 
#     This model is a dynamic feedforward neural network for tabular data. It can take any number of numeric or categorical (input)features dynamically, process categorical features via embeddings, combine them with numeric features, pass them through a configurable MLP, and predict multiple targets simultaneously.
#     Highly configurable: number of layers, hidden dimensions, dropout, batch normalization.
#     This supports multi-target regression or multi-label classification, depending on the loss function used.
#     """
#     def __init__(self, emb_sizes: list[tuple[int, int]], n_numeric: int, n_targets: int, hidden_dims: list[int], dropout: float) -> None:
#         """
#         emb_sizes: List of (num_categories, embedding_size) for each categorical feature.
#         Example: [(5, 3), (10, 4)] → 2 categorical features, first with 5 categories embedded into 3-dim vector, second 10 categories into 4-dim.
#         n_numeric: Number of numeric features.
#         n_targets: Number of outputs the model predicts (multi-target support).
#         hidden_dims: List of hidden layer sizes for the MLP. Example: [128, 64].
#         dropout: Dropout probability.
#         """
#         super().__init__()
#         self.embeddings: nn.ModuleList = nn.ModuleList([nn.Embedding(c, s) for c, s in emb_sizes])
#         # Total dimension after concatenating all embeddings.
#         n_emb: int = sum(s for c, s in emb_sizes)
        
#         layers: list[nn.Module] = []
#         in_dim: int = n_emb + n_numeric
#         for h_dim in hidden_dims:
#             layers.append(nn.Linear(in_dim, h_dim)) # Fully connect layer
#             layers.append(nn.BatchNorm1d(h_dim))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout))
#             in_dim = h_dim
        
#         # The hidden layers are combined in nn.Sequential. output_layer maps the final hidden layer to n_targets outputs.
#         self.network: nn.Sequential = nn.Sequential(*layers) 
#         self.output_layer: nn.Linear = nn.Linear(in_dim, n_targets)
        

#     def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
#         x_emb: list[torch.Tensor] = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
#         x: torch.Tensor = torch.cat(x_emb + [x_num], dim=1)
#         return self.output_layer(self.network(x))

# # # src/my_ml_project/models/train.py

# # import torch
# # from torch import nn
# # from torch.utils.data import DataLoader

# # def train_model(
# #     model: nn.Module,
# #     dataset,
# #     *,
# #     epochs: int,
# #     batch_size: int,
# #     lr: float,
# #     device: str,
# # ):
# #     model.to(device)
# #     model.train()

# #     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# #     criterion = nn.CrossEntropyLoss()

# #     for epoch in range(epochs):
# #         total_loss = 0.0

# #         for x, y in loader:
# #             x, y = x.to(device), y.to(device)

# #             optimizer.zero_grad()
# #             logits = model(x)
# #             loss = criterion(logits, y)
# #             loss.backward()
# #             optimizer.step()

# #             total_loss += loss.item()

# #         print(f"Epoch {epoch + 1}/{epochs} | loss={total_loss:.4f}")

# #     return model


# # Pure PyTorch training logic
# # No Hydra. No file paths. No saving.
# # Reusable
# # Unit-testable
# # Framework-agnostic pipeline