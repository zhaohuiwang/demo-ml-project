

# project-root(demo-ml-project)/src/demo_ml_project/data/dataset.py

# src/demo_ml_project/data/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd


class InputDataset(Dataset):
    """
    PyTorch Dataset for tabular data with separate categorical and numerical features.

    Expects preprocessed data (encoded categoricals, scaled numerics/targets).
    Returns tuples of (categorical indices, numerical features, targets).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cat_cols: list[str],
        num_cols: list[str],
        target_cols: list[str],
    ) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed DataFrame with encoded/scaled columns
        cat_cols : list[str]
            Names of categorical columns (already ordinal-encoded)
        num_cols : list[str]
            Names of numerical feature columns (already scaled)
        target_cols : list[str]
            Names of target columns (already scaled if regression)
        """
        self.cat_features = torch.tensor(
            df[cat_cols].values, dtype=torch.long
        )
        self.num_features = torch.tensor(
            df[num_cols].values, dtype=torch.float32
        )
        self.targets = torch.tensor(
            df[target_cols].values, dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a single sample as (categorical tensor, numerical tensor, target tensor).
        """
        return (
            self.cat_features[idx],
            self.num_features[idx],
            self.targets[idx],
        )
    
# import torch
# from torch.utils.data import Dataset
# import pandas as pd

# class InputDataset(Dataset):
#     def __init__(self, df: pd.DataFrame, cats: list[str], nums: list[str], targets: list[str]) -> None:
#         self.cats: torch.Tensor = torch.tensor(df[cats].values, dtype=torch.long)
#         self.nums: torch.Tensor = torch.tensor(df[nums].values, dtype=torch.float32)
#         self.y: torch.Tensor = torch.tensor(df[targets].values, dtype=torch.float32)

#     def __len__(self) -> int:
#         return len(self.y)

#     def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         return self.cats[idx], self.nums[idx], self.y[idx]