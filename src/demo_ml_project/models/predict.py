

# src/my_ml_project/models/predict.py

import torch
from torch import nn
from torch.utils.data import DataLoader

@torch.no_grad()
def predict(
    model: nn.Module,
    dataset,
    *,
    batch_size: int,
    device: str,
):
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []

    for x in loader:
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())

    return torch.cat(all_preds)


# Reusable
# Testable
# Safe (@torch.no_grad())