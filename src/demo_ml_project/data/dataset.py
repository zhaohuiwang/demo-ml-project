

# project-root(demo-ml-project)/src/demo_ml_project/data/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd

class InputDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cats: list[str], nums: list[str], targets: list[str]) -> None:
        self.cats: torch.Tensor = torch.tensor(df[cats].values, dtype=torch.long)
        self.nums: torch.Tensor = torch.tensor(df[nums].values, dtype=torch.float32)
        self.y: torch.Tensor = torch.tensor(df[targets].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.cats[idx], self.nums[idx], self.y[idx]