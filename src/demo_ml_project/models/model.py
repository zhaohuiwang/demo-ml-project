
# project-root(demo-ml-project)/src/demo_ml_project/models/model.py
# src/demo_ml_project/models/model.py

# import torch
# import torch.nn as nn
# from typing import List, Optional, Union


# class TabularModel(nn.Module):
#     """
#     Modern tabular model that takes ONE input tensor x of shape (batch_size, total_features).
    
#     - First `n_cat_features` columns → treated as categorical (integer indices)
#     - Remaining columns → treated as numerical (float)
    
#     Supports:
#     - Mixed categorical + numerical features
#     - Multi-target regression / multi-label / multi-class
#     - Learnable numerical normalization
#     - Flexible activations, normalization, initialization
#     """
    
#     def __init__(
#         self,
#         cat_cardinalities: List[int],               # num categories for each categorical feature
#         embedding_dims: List[int],                  # embedding dim for each categorical feature
#         n_num_features: int,
#         n_targets: int,
#         hidden_dims: List[int],
#         dropout: Union[float, List[float]] = 0.1,
#         use_batchnorm: bool = True,
#         use_layernorm: bool = False,
#         activation: str = "relu",                   # "gelu", "gelu", "mish", ...
#         numerical_normalization: bool = True,
#         output_constraint: Optional[str] = None,    # None | "sigmoid" | "softplus" | "tanh"
#         init_method: str = "kaiming",
#     ):
#         super().__init__()

#         assert len(cat_cardinalities) == len(embedding_dims)
#         assert not (use_batchnorm and use_layernorm), "Choose BatchNorm OR LayerNorm"

#         self.n_cat = len(cat_cardinalities)
#         self.n_num = n_num_features
#         self.total_in_features = self.n_cat + self.n_num

#         # ─── Embeddings ───────────────────────────────────────────────────────
#         self.embeddings = nn.ModuleList([
#             nn.Embedding(num_embeddings=card + 1, embedding_dim=dim)
#             for card, dim in zip(cat_cardinalities, embedding_dims)
#         ])

#         total_emb_dim = sum(embedding_dims)

#         # ─── Numerical normalizer ─────────────────────────────────────────────
#         self.num_normalizer = None
#         if numerical_normalization and self.n_num > 0:
#             self.num_normalizer = nn.BatchNorm1d(
#                 self.n_num,
#                 affine=True,
#                 track_running_stats=False   # common choice for tabular data
#             )

#         # ─── Backbone input dim ───────────────────────────────────────────────
#         backbone_in_dim = total_emb_dim + self.n_num

#         # ─── MLP backbone ─────────────────────────────────────────────────────
#         layers = []
#         current_dim = backbone_in_dim

#         dropout_rates = [dropout] * len(hidden_dims) if isinstance(dropout, float) else dropout
#         assert len(dropout_rates) == len(hidden_dims)

#         for hdim, dr in zip(hidden_dims, dropout_rates):
#             linear = nn.Linear(current_dim, hdim)

#             # Initialization
#             if init_method == "kaiming":
#                 nn.init.kaiming_normal_(linear.weight, nonlinearity=activation)
#             elif init_method == "xavier":
#                 nn.init.xavier_uniform_(linear.weight)

#             layers.append(linear)

#             if use_batchnorm:
#                 layers.append(nn.BatchNorm1d(hdim))
#             if use_layernorm:
#                 layers.append(nn.LayerNorm(hdim))

#             if activation == "relu":
#                 layers.append(nn.ReLU())
#             elif activation == "gelu":
#                 layers.append(nn.GELU())
#             elif activation == "mish":
#                 layers.append(nn.Mish())
#             else:
#                 raise ValueError(f"Unknown activation: {activation}")

#             layers.append(nn.Dropout(dr))
#             current_dim = hdim

#         self.backbone = nn.Sequential(*layers)

#         # ─── Head ─────────────────────────────────────────────────────────────
#         self.head = nn.Linear(current_dim, n_targets)

#         # Optional output activation
#         self.output_act = None
#         if output_constraint == "sigmoid":
#             self.output_act = nn.Sigmoid()
#         elif output_constraint == "softplus":
#             self.output_act = nn.Softplus()
#         elif output_constraint == "tanh":
#             self.output_act = nn.Tanh()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x : (batch_size, n_cat + n_num)
#             First n_cat columns: integer indices (long)
#             Next  n_num columns: float values (already preprocessed/scaled)
#         """
#         assert x.shape[1] == self.total_in_features, \
#             f"Expected {self.total_in_features} features, got {x.shape[1]}"

#         # Split input
#         x_cat = x[:, :self.n_cat].long()      # (batch, n_cat)
#         x_num = x[:, self.n_cat:].float()     # (batch, n_num)

#         # ─── Categorical embeddings ───────────────────────────────────────────
#         if self.n_cat > 0:
#             embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
#             x_emb = torch.cat(embs, dim=1)          # (batch, total_emb_dim)
#         else:
#             x_emb = torch.empty(x.shape[0], 0, device=x.device, dtype=x.dtype)

#         # ─── Numerical part ───────────────────────────────────────────────────
#         if self.n_num > 0:
#             x_n = x_num
#             if self.num_normalizer is not None:
#                 x_n = self.num_normalizer(x_n)
#         else:
#             x_n = torch.empty(x.shape[0], 0, device=x.device, dtype=x.dtype)

#         # ─── Concat & backbone ────────────────────────────────────────────────
#         x = torch.cat([x_emb, x_n], dim=1)
#         x = self.backbone(x)

#         # ─── Output ───────────────────────────────────────────────────────────
#         out = self.head(x)
#         if self.output_act is not None:
#             out = self.output_act(out)

#         return out


# # ─── Example instantiation ───────────────────────────────────────────────────
# def example_model():
#     return TabularModel(
#         cat_cardinalities = [20, 7, 150, 5],
#         embedding_dims    = [8, 5, 12, 4],
#         n_num_features    = 28,
#         n_targets         = 4,               # e.g. 4 regression targets
#         hidden_dims       = [512, 256, 128],
#         dropout           = 0.12,
#         use_batchnorm     = True,
#         activation        = "relu",
#         numerical_normalization = True,
#         output_constraint = None,
#         init_method       = "kaiming"
#     )

# model = example_model().cuda()
# x = torch.cat([cat_tensor.long(), num_tensor.float()], dim=1).cuda()   # shape (bs, 20+7+150+5 + 28)
# pred = model(x)   # shape (bs, 4)




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
            
        # Fully connected multi-Layer Perceptron.
        self.mlp = nn.Sequential(*layers) 

        # Final output layer
        self.output_head = nn.Linear(current_dim, n_targets)

    def forward(
        self,
        x: torch.Tensor,
        # x_cat: torch.Tensor,    # shape: (batch_size, num_cat_features)
        # x_num: torch.Tensor,    # shape: (batch_size, n_numeric)
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

