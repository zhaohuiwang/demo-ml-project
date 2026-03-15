import torch
from torch.utils.data import Dataset
import pandas as pd


class InputDataset(Dataset):
    """
    Custom Dataset for tabular data with categorical + numerical features and multi-target regression.
    
    Args:
        df (pd.DataFrame): Input DataFrame with all required columns
        cat_cols (list[str]): Names of categorical columns (will be converted to long/int64)
        num_cols (list[str]): Names of numerical columns (float32)
        target_cols (list[str]): Names of target columns (float32)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        cat_cols: list[str],
        num_cols: list[str],
        target_cols: list[str],
    ):
        self.df = df.reset_index(drop=True)  # ensure consistent indexing
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_cols = target_cols

        # Optional safety check (recommended)
        required = cat_cols + num_cols + target_cols
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Categorical features → long tensor (for embedding layers)
        x_cat = (
            torch.tensor(row[self.cat_cols].values, dtype=torch.long)
            if self.cat_cols else torch.empty(0, dtype=torch.long)
        )

        # Numerical features → float32
        x_num = (
            torch.tensor(row[self.num_cols].values, dtype=torch.float32)
            if self.num_cols else torch.empty(0, dtype=torch.float32)
        )

        # Targets (multi-output regression)
        y = (
            torch.tensor(row[self.target_cols].values, dtype=torch.float32)
            if self.target_cols else torch.empty(0, dtype=torch.float32)
        )

        return x_cat, x_num, y