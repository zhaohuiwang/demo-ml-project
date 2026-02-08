# src/demo_ml_project/pipelines/model_training.py
import pandas as pd
from pathlib import Path
import json
import joblib
import torch
import optuna
from torch.utils.data import DataLoader
from ..configs.schema import RootConfig

from ..configs.schema import RootConfig
from ..configs.artifacts import (
    PreprocessingArtifacts,
    TrainingArtifacts
    )

from ..data.processing import prepare_data
from ..data.dataset import InputDataset
from ..models.model import DynamicModel
from ..optimization.objective import objective
from ..utils.logging import get_logger

# src/demo_ml_project/pipelines/model_training.py
class TrainingPipeline:
    def __init__(self, cfg: RootConfig):
        self.cfg = cfg
        self.logger = get_logger(self.__class__.__name__)
        self.device = torch.device(self.cfg.device)

        # Access is now type-safe and IDE-friendly
        # self.cfg.training.batch_size     → int
        # self.cfg.optuna.n_trials         → int
        # self.cfg.data.target_cols        → List[str]


        self._preprocessing_artifacts: PreprocessingArtifacts | None = None
        self._val_loader: DataLoader | None = None
        self._export_dir: Path | None = None

    def run(self) -> None:
        artifacts = self.train_only()

        self._export_dir = Path(self.cfg.export.dir)
        self._export_dir.mkdir(parents=True, exist_ok=True)

        metrics = self.evaluate(artifacts.model)
        self._export(artifacts, metrics)

        self.logger.info("Full training pipeline completed ✓")

    def train_only(self) -> TrainingArtifacts:
        df = self._load_and_clean_data()
        self._preprocessing_artifacts = prepare_data(df, self.cfg)
        df_processed = self._preprocessing_artifacts.processed_df

        preprocessors = {
            "cat_encoder": self._preprocessing_artifacts.cat_encoder,
            "num_scaler": self._preprocessing_artifacts.num_scaler,
            "tar_scaler": self._preprocessing_artifacts.tar_scaler,
        }

        emb_sizes = self._compute_embedding_sizes(df_processed, self.cfg.data.cat_cols)

        train_loader, val_loader = self._create_dataloaders(df_processed)
        self._val_loader = val_loader

        study = self._optimize_hyperparameters(train_loader, val_loader, emb_sizes)

        champion = self._instantiate_best_model(study.best_params, emb_sizes)

        return TrainingArtifacts(
            model=champion,
            study=study,
            emb_sizes=emb_sizes,
            preprocessors=preprocessors,
        )

    # ──────────────────────────────────────────────
    #   Most methods remain very similar — just change cfg → self.cfg
    # ──────────────────────────────────────────────
    def _load_and_clean_data(self) -> pd.DataFrame:
        path = Path(self.cfg.data.train_data_path)
        if not path.exists():
            raise FileNotFoundError(f"Training data not found: {path}")

        df = pd.read_parquet(path)
        if self.cfg.data.drop_columns:
            df = df.drop(columns=self.cfg.data.drop_columns, errors="ignore")

        # fillna logic ...
        return df

    def _create_dataloaders(self, df: pd.DataFrame) -> tuple[DataLoader, DataLoader]:
        from sklearn.model_selection import train_test_split

        train_df, val_df = train_test_split(
            df,
            test_size=self.cfg.training.test_size,
            random_state=self.cfg.training.random_state,
            shuffle=True,
        )

        train_ds = InputDataset(train_df, self.cfg.data.cat_cols, self.cfg.data.num_cols, self.cfg.data.target_cols)
        val_ds = InputDataset(val_df, self.cfg.data.cat_cols, self.cfg.data.num_cols, self.cfg.data.target_cols)

        common_kw = dict(
            batch_size=self.cfg.training.batch_size,
            num_workers=2,
            pin_memory=self.device.type != "cpu",
            persistent_workers=True,
        )

        return (
            DataLoader(train_ds, shuffle=True, **common_kw),
            DataLoader(val_ds, shuffle=False, **common_kw),
        )

    def _optimize_hyperparameters(self, train_loader, val_loader, emb_sizes):
        # pass self.cfg to objective
        study.optimize(
            lambda trial: objective(trial, self.cfg, train_loader, val_loader, emb_sizes, self.device),
            n_trials=self.cfg.optuna.n_trials,
            show_progress_bar=True,
        )
        return study

    # _instantiate_best_model, evaluate, _export remain almost identical
    # Just use self.cfg.export.xxx instead of self.cfg.export.xxx