
# src/demo_ml_project/__init__.py

"""
demo-ml-project
===============

A configurable PyTorch-based tabular regression pipeline with:
- Hydra configuration
- Optuna hyperparameter optimization (with optional CV)
- Leakage-safe preprocessing
- Multi-target support
- Early stopping & model export

Main entry point: scripts/train.py
"""

__version__ = "0.1.0"

# Optional: expose the most frequently used classes
from .configs.schema import RootConfig
from .pipelines.model_training import TrainingPipeline
from .models.model import DynamicTabularModel
from .data.processing import prepare_data
from .data.dataset import InputDataset

__all__ = [
    "RootConfig",
    "TrainingPipeline",
    "DynamicTabularModel",
    "prepare_data",
    "InputDataset",
]