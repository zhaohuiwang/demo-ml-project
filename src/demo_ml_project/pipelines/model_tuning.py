
import torch.optim as optim
import torch
from torch import nn
from typing import Dict, Any

from my_ml_project.pipelines.basepipeline import BasePipeline
from my_ml_project.pipelines.model_tuning_utils import objective

class TuningPipeline(BasePipeline):
    def __init__(
        self,
        cfg: ConfigSchema,
        train_loader: DataLoader,
        val_loader: DataLoader,
        emb_sizes: list[tuple[int, int]],
        device: torch.device,
    ):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.emb_sizes = emb_sizes
        self.device = device

    def run(self) -> optuna.Study:
        study = optuna.create_study(direction="minimize")

        study.optimize(
            lambda t: objective(
                t,
                self.cfg,
                self.train_loader,
                self.val_loader,
                self.emb_sizes,
                self.device,
            ),
            n_trials=self.cfg.optuna.n_trials,
        )

        return study
