
# project-root(demo-ml-project)/src/demo_ml_project/pipelines/model_training.py

from __future__ import annotations


import json
from pathlib import Path

import joblib
import optuna
import pandas as pd
import torch
from datetime import datetime
from torch.utils.data import DataLoader

from ..configs.schema import (
    ConfigSchema,
    PreprocessingArtifacts,
    TrainingArtifacts
    )
from ..data.processing import prepare_data
from ..data.dataset import InputDataset
from ..models.model import DynamicModel
from ..optimization.objective import objective
from ..utils.logging import get_logger



class TrainingPipeline:
    def __init__(self, cfg: ConfigSchema):
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
        path = self.cfg.data.train_data_path
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
            num_workers=2,              # ← SUGGESTION
            pin_memory=self.device.type != "cpu",
            persistent_workers=True,
        )

        return (
            DataLoader(train_ds, shuffle=True, **common_kw),
            DataLoader(val_ds, shuffle=False, **common_kw),
        )

    def _optimize_hyperparameters(self, train_loader, val_loader, emb_sizes):
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner, HyperbandPruner

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

        (p / self.cfg.export.metadata).write_text(json.dumps(metadata, indent=2))


# from __future__ import annotations

# import json
# import joblib
# import torch
# import optuna
# import pandas as pd

# from pathlib import Path
# from typing import Any, Dict
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader

# from demo_ml_project.configs.schema import ConfigSchema
# from demo_ml_project.data.processing import prepare_data
# from demo_ml_project.data.dataset import InputDataset
# from demo_ml_project.models.model import DynamicModel
# from demo_ml_project.optimization.objective import objective
# from demo_ml_project.utils.logging import get_logger


# class TrainingPipeline:
#     """
#     Minimal, extensible training pipeline.
#     """

#     def __init__(self, cfg: ConfigSchema, export_dir: Path):
#         self.cfg = cfg
#         self.export_dir = export_dir
#         self.logger = get_logger(self.__class__.__name__)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # cached artifacts
#         self._df_proc: pd.DataFrame | None = None
#         self._val_loader: DataLoader | None = None

#     # ======================================================
#     # Public API
#     # ======================================================

#     def run(self) -> None:
#         """
#         Full pipeline: train + evaluate + export.
#         """
#         self.logger.info("Running full training pipeline")

#         artifacts = self.train_only()
#         metrics = self.evaluate(artifacts["model"])

#         self._export(
#             model=artifacts["model"],
#             study=artifacts["study"],
#             emb_sizes=artifacts["emb_sizes"],
#             preprocessors=artifacts["preprocessors"],
#             metrics=metrics,
#         )

#         self.logger.info("Pipeline finished successfully")

#     def train_only(self) -> Dict[str, Any]:
#         """
#         Train the model but do NOT export or evaluate.
#         Useful for hyperparameter sweeps, notebooks, tests.
#         """
#         self.logger.info("Starting training only")

#         df = self._load_dataframe()
#         df_proc, preprocessors = self._prepare_features(df)
#         emb_sizes = self._build_embeddings(df_proc)

#         train_loader, val_loader = self._build_dataloaders(df_proc)
#         self._df_proc = df_proc
#         self._val_loader = val_loader

#         study = self._run_optuna(train_loader, val_loader, emb_sizes)
#         model = self._build_champion_model(study.best_params, emb_sizes)

#         return {
#             "model": model,
#             "study": study,
#             "emb_sizes": emb_sizes,
#             "preprocessors": preprocessors,
#         }

#     def evaluate(self, model: DynamicModel) -> Dict[str, float]:
#         """
#         Evaluate a trained model on the validation set.
#         """
#         if self._val_loader is None:
#             raise RuntimeError("evaluate() called before train_only()")

#         self.logger.info("Evaluating model")

#         model.eval()
#         total_loss = 0.0
#         n_batches = 0

#         criterion = torch.nn.MSELoss()  # adjust if needed

#         with torch.no_grad():
#             for batch in self._val_loader:
#                 x_cat, x_num, y = batch
#                 x_cat = x_cat.to(self.device)
#                 x_num = x_num.to(self.device)
#                 y = y.to(self.device)

#                 preds = model(x_cat, x_num)
#                 loss = criterion(preds, y)
#                 total_loss += loss.item()
#                 n_batches += 1

#         metrics = {
#             "val_loss": total_loss / max(n_batches, 1)
#         }

#         self.logger.info(f"Evaluation metrics: {metrics}")
#         return metrics

#     # ======================================================
#     # Internal steps
#     # ======================================================

#     def _load_dataframe(self) -> pd.DataFrame:
#         df = pd.read_parquet(self.cfg.data.train_data_path).drop(columns=self.cfg.data.drop_columns)
#         if df.empty:
#             raise ValueError("Dataframe is empty after preprocessing")

#         df["population"] = df["population"].fillna(df["population"].median()).astype(int)
#         df = df.fillna(df.median(numeric_only=True))
#         return df

#     def _prepare_features(self, df: pd.DataFrame):
#         df_proc, cat_encoder, num_scaler, tar_scaler = prepare_data(df, self.cfg)
#         return df_proc, {
#             "cat_encoder": cat_encoder,
#             "num_scaler": num_scaler,
#             "tar_scaler": tar_scaler,
#         }

#     def _build_embeddings(self, df_proc: pd.DataFrame):
#         return [
#             (int(df_proc[col].nunique()), min(50, (int(df_proc[col].nunique()) + 1) // 2))
#             for col in self.cfg.data.cat_cols
#         ]

#     def _build_dataloaders(self, df_proc: pd.DataFrame):
#         train_df, val_df = train_test_split(
#             df_proc,
#             test_size=self.cfg.training.test_size,
#             random_state=self.cfg.training.random_state,
#         )

#         train_loader = DataLoader(
#             InputDataset(train_df, self.cfg.data.cat_cols, self.cfg.data.num_cols, self.cfg.data.target_cols),
#             batch_size=self.cfg.training.batch_size,
#             shuffle=True,
#         )

#         val_loader = DataLoader(
#             InputDataset(val_df, self.cfg.data.cat_cols, self.cfg.data.num_cols, self.cfg.data.target_cols),
#             batch_size=self.cfg.training.batch_size,
#         )

#         return train_loader, val_loader

#     def _run_optuna(self, train_loader, val_loader, emb_sizes):
#         self.logger.info("Starting Optuna optimization")

#         study = optuna.create_study(direction="minimize")
#         study.optimize(
#             lambda t: objective(
#                 t,
#                 self.cfg,
#                 train_loader,
#                 val_loader,
#                 emb_sizes,
#                 self.device,
#             ),
#             n_trials=self.cfg.optuna.n_trials,
#         )

#         self.logger.info(f"Optuna finished. Best params: {study.best_params}")
#         return study

#     def _build_champion_model(self, best_params: dict[str, Any], emb_sizes):
#         hidden_dims = [
#             best_params[f"n_units_l{i}"]
#             for i in range(best_params["n_layers"])
#         ]

#         return DynamicModel(
#             emb_sizes=emb_sizes,
#             n_numeric=len(self.cfg.data.num_cols),
#             n_targets=len(self.cfg.data.target_cols),
#             hidden_dims=hidden_dims,
#             dropout=best_params["dropout"],
#         ).to(self.device)

#     def _export(
#         self,
#         model: DynamicModel,
#         study: optuna.Study,
#         emb_sizes,
#         preprocessors,
#         metrics: Dict[str, float],
#     ) -> None:
#         self.export_dir.mkdir(parents=True, exist_ok=True)

#         torch.save(model.state_dict(), self.export_dir / self.cfg.export.weights)

#         joblib.dump(preprocessors["num_scaler"], self.export_dir / self.cfg.export.in_scaler)
#         joblib.dump(preprocessors["tar_scaler"], self.export_dir / self.cfg.export.tar_scaler)
#         joblib.dump(preprocessors["cat_encoder"], self.export_dir / self.cfg.export.cat_encoder)

#         metadata = {
#             "best_params": study.best_params,
#             "emb_sizes": emb_sizes,
#             "num_cols": self.cfg.data.num_cols,
#             "cat_cols": self.cfg.data.cat_cols,
#             "target_names": self.cfg.data.target_cols,
#             "metrics": metrics,
#         }

#         with open(self.export_dir / self.cfg.export.metadata, "w") as f:
#             json.dump(metadata, f, indent=4)


# # src/my_ml_project/pipelines/training_pipeline.py

# import torch
# from torch import nn
# from my_ml_project.pipelines.basepipeline import BasePipeline
# from my_ml_project.data.loaders import load_torch_dataset
# from my_ml_project.models.train import train_model
# from my_ml_project.registry.factory import get_model_registry

# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, output_dim),
#         )

#     def forward(self, x):
#         return self.net(x)



# class TrainingPipeline(BasePipeline):
#     def run_training_pipeline(cfg):
#         # Reproducibility
#         torch.manual_seed(cfg.seed)

#         # Dataset
#         dataset = load_torch_dataset(cfg.data.train_path)

#         # Model
#         model = MLP(
#             input_dim=cfg.model.input_dim,
#             hidden_dim=cfg.model.hidden_dim,
#             output_dim=cfg.model.output_dim,
#             dropout=cfg.model.dropout,
#         )

#         # Train
#         trained_model = train_model(
#             model=model,
#             dataset=dataset,
#             epochs=cfg.training.epochs,
#             batch_size=cfg.training.batch_size,
#             lr=cfg.training.lr,
#             device=cfg.training.device,
#         )

#         # Save via Registry
#         registry = get_model_registry(cfg)

#         registry.save(
#             model=trained_model,
#             name=cfg.project_name,
#         )

#         return trained_model
    
#     def run(self) -> None:
#         pass

# Central place for wiring
# Easy to swap models
# Easy to integrate MLflow later


# How Feast connects to training (offline features)
# Your training pipeline today
# X, y = build_features_from_sql(...)
# model.fit(X, y)
# With Feast
# from feast import FeatureStore

# store = FeatureStore(repo_path="feature-repo")

# training_df = store.get_historical_features(
#     entity_df=entity_df,
#     features=[
#         "user_stats:ctr_7d",
#         "user_stats:click_count_30d",
#     ],
# ).to_df()

# What changed?
# You no longer compute features in training
# You ask Feast for point-in-time correct features
# Same feature definitions are used later in inference

# How Feast connects to inference (online features)
# In your FastAPI service

# store = FeatureStore(repo_path="feature-repo")

# features = store.get_online_features(
#     features=[
#         "user_stats:ctr_7d",
#         "user_stats:click_count_30d",
#     ],
#     entity_rows=[{"user_id": user_id}],
# ).to_dict()
# training and inference pull identical feature names