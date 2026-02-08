
#  Local Filesystem Registry (default)

import torch
from pathlib import Path
from my_ml_project.registry.base import ModelRegistry

class LocalModelRegistry(ModelRegistry):
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)

    def save(self, model, name: str, version: str) -> None:
        path = self.root / name / version
        path.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), path / "model.pt")

    def load(self, name: str, version: str):
        path = self.root / name / version / "model.pt"
        return torch.load(path, map_location="cpu")


# Directory layout:
# model_registry/
# └── classifier/
#     └── v1/
#         └── model.pt
