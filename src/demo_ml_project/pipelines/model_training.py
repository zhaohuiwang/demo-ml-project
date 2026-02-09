
# project-root(demo-ml-project)/src/demo_ml_project/pipelines/model_training.py

import json
from pathlib import Path

import joblib
import optuna
import pandas as pd
import torch

from datetime import datetime
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner

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


class TrainingPipeline:
    def __init__(self, cfg: RootConfig):
        self.cfg = cfg
        self.logger = get_logger(self.__class__.__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # Cache expensive objects
        self._preprocessing_artifacts: PreprocessingArtifacts | None = None
        self._val_loader: DataLoader | None = None
        self._export_dir: Path | None = None

    def run(self) -> None:
        artifacts = self.train_only()

        self._export_dir = Path(self.cfg.export.dir)
        self._export_dir.mkdir(parents=True, exist_ok=True)

        metrics = self.evaluate(artifacts.model)
        self.logger.info(f"Validation metrics: {metrics}")

        self._export(artifacts, metrics)
        self.logger.info("Full training pipeline completed ✓")

    def train_only(self) -> TrainingArtifacts:
        if self._preprocessing_artifacts is None:
            df = self._load_and_clean_data()
            self._preprocessing_artifacts=prepare_data(df, self.cfg)
            df_processed = self._preprocessing_artifacts.processed_df
            preprocessors = {
                "cat_encoder": self._preprocessing_artifacts.cat_encoder,
                "num_scaler": self._preprocessing_artifacts.num_scaler,
                "tar_scaler": self._preprocessing_artifacts.tar_scaler,
                }

        else:
            preprocessors = {}  # shouldn't happen in normal flow

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

    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        if self._val_loader is None:
            raise RuntimeError("Cannot evaluate before creating validation loader")

        self.logger.info("Running validation pass...")

        model.eval()
        criterion = torch.nn.MSELoss()
        total_loss = 0.0
        n = 0

        with torch.inference_mode():
            for x_cat, x_num, y_true in self._val_loader:
                x_cat = x_cat.to(self.device, non_blocking=True)
                x_num = x_num.to(self.device, non_blocking=True)
                y_true = y_true.to(self.device, non_blocking=True)

                pred = model(x_cat, x_num)
                loss = criterion(pred, y_true)

                total_loss += loss.item() * len(y_true)
                n += len(y_true)

        return {"val_rmse": (total_loss / n) ** 0.5, "val_mse": total_loss / n}

    # ──────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────

    def _load_and_clean_data(self) -> pd.DataFrame:
        path = Path(self.cfg.data.train_data_path)
        if not path or not path.exists():
            raise FileNotFoundError(f"Training data not found: {path}")

        df = pd.read_parquet(path)
        if self.cfg.data.drop_columns:
            df = df.drop(columns=self.cfg.data.drop_columns, errors="ignore")

        # Domain-specific fillna → move to config or separate cleaning step in production
        if "population" in df.columns:
            df["population"] = df["population"].fillna(df["population"].median()).astype("int32")

        df = df.fillna(df.select_dtypes(include="number").median())
        return df

    @staticmethod
    def _compute_embedding_sizes(df: pd.DataFrame, cat_cols: list[str]) -> list[tuple[int, int]]:
        return [
            (int(df[col].nunique()), min(60, (int(df[col].nunique()) + 1) // 2))
            for col in cat_cols
        ]

    def _create_dataloaders(self, df: pd.DataFrame) -> tuple[DataLoader, DataLoader]:
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
            num_workers=2,              # ← SUGGESTION
            pin_memory=self.device.type != "cpu",
            persistent_workers=True,
        )

        return (
            DataLoader(train_ds, shuffle=True, **common_kw),
            DataLoader(val_ds, shuffle=False, **common_kw),
        )

    def _optimize_hyperparameters(self, train_loader, val_loader, emb_sizes):
        pruner_map = {"median": MedianPruner(), "hyperband": HyperbandPruner(), "nop": optuna.pruners.NopPruner()}

        sampler_map = {
            "tpe": TPESampler(multivariate=True, group=True),
            # "random": optuna.samplers.RandomSampler(),
            # "cmaes": optuna.samplers.CmaEsSampler(),
        }

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler_map.get(self.cfg.optuna.sampler, TPESampler()),
            pruner=pruner_map.get(self.cfg.optuna.pruner, MedianPruner()),
            storage=None,  # can be "sqlite:///optuna.db" later
        )

        study.optimize(
            lambda t: objective(t, self.cfg, train_loader, val_loader, emb_sizes, self.device),
            n_trials=self.cfg.optuna.n_trials,
            show_progress_bar=True,           # ← nice for longer runs
        )

        self.logger.info("Best value: %.5f", study.best_value)
        self.logger.info("Best params: %s", study.best_params)

        return study

    def _instantiate_best_model(self, best_params: dict, emb_sizes):
        hidden_dims = [best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])]

        return DynamicModel(
            emb_sizes=emb_sizes,
            n_numeric=len(self.cfg.data.num_cols),
            n_targets=len(self.cfg.data.target_cols),
            hidden_dims=hidden_dims,
            dropout=best_params["dropout"],
        ).to(self.device)

    def _export(self, artifacts: TrainingArtifacts, metrics: dict[str, float]):
        if self._export_dir is None:
            raise RuntimeError("Export directory not set")

        p = self._export_dir

        torch.save(artifacts.model.state_dict(), p / self.cfg.export.weights)

        joblib.dump(artifacts.preprocessors["num_scaler"], p / self.cfg.export.in_scaler)
        joblib.dump(artifacts.preprocessors["tar_scaler"], p / self.cfg.export.tar_scaler)
        joblib.dump(artifacts.preprocessors["cat_encoder"], p / self.cfg.export.cat_encoder)


        metadata = {
            "best_params": artifacts.study.best_params,
            "best_value": artifacts.study.best_value,
            "emb_sizes": artifacts.emb_sizes, 
            "numeric_features": self.cfg.data.num_cols,
            "categorical_features": self.cfg.data.cat_cols,
            "targets": self.cfg.data.target_cols,
            "val_metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            }
        
        (p / self.cfg.export.metadata).write_text(
            json.dumps(metadata, indent=2)
            )
