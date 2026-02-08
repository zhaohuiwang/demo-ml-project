# Final public API
# pipeline = TrainingPipeline(cfg, export_dir)

# pipeline.run()                      # train + eval + export
# pipeline.train_only()               # train only
# pipeline.evaluate()                 # load model + eval
# pipeline.evaluate(model=my_model)   # eval in-memory model
# pipeline.predict(df)                # inference
# pipeline.resume_from_checkpoint()   # resume training

# Update from model_training.py
# full pipeline
# train-only
# evaluate() works standalone
# metrics are pluggable
# resume training is one flag
# predict() enables inference pipelines

from __future__ import annotations

import json
import joblib
import torch
import optuna
import pandas as pd

from pathlib import Path
from typing import Any, Dict, Callable, Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from demo_ml_package.configs.schema import ConfigSchema
from demo_ml_package.data.processing import prepare_data
from demo_ml_package.data.dataset import InputDataset
from demo_ml_package.models.model import DynamicModel
from demo_ml_package.optimization.objective import objective
from demo_ml_package.utils.logging import get_logger


MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


class TrainingPipeline:
    """
    Minimal but complete training + evaluation + inference pipeline.
    """

    def __init__(self, cfg: ConfigSchema, export_dir: Path):
        self.cfg = cfg
        self.export_dir = export_dir
        self.logger = get_logger(self.__class__.__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._val_loader: DataLoader | None = None
        self._metrics: Dict[str, MetricFn] = self._default_metrics()

    # ======================================================
    # Public API
    # ======================================================

    def run(self) -> None:
        artifacts = self.train_only()
        metrics = self.evaluate(model=artifacts["model"])

        self._export(
            model=artifacts["model"],
            study=artifacts["study"],
            emb_sizes=artifacts["emb_sizes"],
            preprocessors=artifacts["preprocessors"],
            metrics=metrics,
        )

    def train_only(self, resume_from_checkpoint: bool = False) -> Dict[str, Any]:
        self.logger.info("Starting training")

        df = self._load_dataframe()
        df_proc, preprocessors = self._prepare_features(df)
        emb_sizes = self._build_embeddings(df_proc)
        train_loader, val_loader = self._build_dataloaders(df_proc)

        self._val_loader = val_loader

        study = self._run_optuna(train_loader, val_loader, emb_sizes)
        model = self._build_champion_model(study.best_params, emb_sizes)

        if resume_from_checkpoint:
            self._load_model_weights(model)

        return {
            "model": model,
            "study": study,
            "emb_sizes": emb_sizes,
            "preprocessors": preprocessors,
        }

    def evaluate(
        self,
        model: Optional[DynamicModel] = None,
        metrics: Optional[Dict[str, MetricFn]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model. Loads model from disk if not provided.
        """
        self.logger.info("Evaluating model")

        if model is None:
            model = self._load_model()

        if self._val_loader is None:
            _, self._val_loader = self._build_dataloaders(
                self._prepare_features(self._load_dataframe())[0]
            )

        metrics = metrics or self._metrics
        results = {k: 0.0 for k in metrics}
        counts = 0

        model.eval()
        with torch.no_grad():
            for x_cat, x_num, y in self._val_loader:
                x_cat, x_num, y = (
                    x_cat.to(self.device),
                    x_num.to(self.device),
                    y.to(self.device),
                )

                preds = model(x_cat, x_num)

                for name, fn in metrics.items():
                    results[name] += fn(preds, y)

                counts += 1

        final_metrics = {k: v / max(counts, 1) for k, v in results.items()}
        self.logger.info(f"Metrics: {final_metrics}")
        return final_metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on new data.
        """
        self.logger.info("Running inference")

        model = self._load_model()
        preprocessors = self._load_preprocessors()

        df_proc = self._apply_preprocessors(df, preprocessors)

        dataset = InputDataset(
            df_proc,
            self.cfg.data.cat_cols,
            self.cfg.data.num_cols,
            self.cfg.data.target_cols,
        )

        loader = DataLoader(dataset, batch_size=self.cfg.training.batch_size)
        preds = []

        model.eval()
        with torch.no_grad():
            for x_cat, x_num, _ in loader:
                x_cat, x_num = x_cat.to(self.device), x_num.to(self.device)
                preds.append(model(x_cat, x_num).cpu())

        return pd.DataFrame(torch.cat(preds).numpy(), columns=self.cfg.data.target_cols)

    # ======================================================
    # Internal helpers
    # ======================================================

    def _default_metrics(self) -> Dict[str, MetricFn]:
        return {
            "mse": lambda p, y: torch.mean((p - y) ** 2).item(),
            "mae": lambda p, y: torch.mean(torch.abs(p - y)).item(),
        }

    def _load_dataframe(self) -> pd.DataFrame:
        df = pd.read_parquet(self.cfg.data.path).drop(columns=self.cfg.data.drop_columns)
        df["population"] = df["population"].fillna(df["population"].median()).astype(int)
        return df.fillna(df.median(numeric_only=True))

    def _prepare_features(self, df: pd.DataFrame):
        return prepare_data(df, self.cfg)

    def _build_embeddings(self, df_proc: pd.DataFrame):
        return [
            (int(df_proc[c].nunique()), min(50, (int(df_proc[c].nunique()) + 1) // 2))
            for c in self.cfg.data.cat_cols
        ]

    def _build_dataloaders(self, df_proc: pd.DataFrame):
        train_df, val_df = train_test_split(
            df_proc,
            test_size=self.cfg.training.test_size,
            random_state=self.cfg.training.random_state,
        )

        return (
            DataLoader(
                InputDataset(train_df, self.cfg.data.cat_cols, self.cfg.data.num_cols, self.cfg.data.target_cols),
                batch_size=self.cfg.training.batch_size,
                shuffle=True,
            ),
            DataLoader(
                InputDataset(val_df, self.cfg.data.cat_cols, self.cfg.data.num_cols, self.cfg.data.target_cols),
                batch_size=self.cfg.training.batch_size,
            ),
        )

    def _run_optuna(self, train_loader, val_loader, emb_sizes):
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda t: objective(t, self.cfg, train_loader, val_loader, emb_sizes, self.device),
            n_trials=self.cfg.optuna.n_trials,
        )
        return study

    def _build_champion_model(self, best_params, emb_sizes):
        hidden_dims = [best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])]
        return DynamicModel(
            emb_sizes=emb_sizes,
            n_numeric=len(self.cfg.data.num_cols),
            n_targets=len(self.cfg.data.target_cols),
            hidden_dims=hidden_dims,
            dropout=best_params["dropout"],
        ).to(self.device)

    def _load_model(self) -> DynamicModel:
        self.logger.info("Loading model from disk")
        with open(self.export_dir / self.cfg.export.metadata) as f:
            meta = json.load(f)

        model = DynamicModel(
            emb_sizes=meta["emb_sizes"],
            n_numeric=len(meta["num_cols"]),
            n_targets=len(meta["target_names"]),
            hidden_dims=[meta["best_params"][f"n_units_l{i}"] for i in range(meta["best_params"]["n_layers"])],
            dropout=meta["best_params"]["dropout"],
        ).to(self.device)

        model.load_state_dict(torch.load(self.export_dir / self.cfg.export.weights))
        return model

    def _load_model_weights(self, model: DynamicModel) -> None:
        path = self.export_dir / self.cfg.export.weights
        if path.exists():
            model.load_state_dict(torch.load(path))

    def _load_preprocessors(self):
        return {
            "num_scaler": joblib.load(self.export_dir / self.cfg.export.in_scaler),
            "tar_scaler": joblib.load(self.export_dir / self.cfg.export.tar_scaler),
            "cat_encoder": joblib.load(self.export_dir / self.cfg.export.cat_encoder),
        }

    def _apply_preprocessors(self, df: pd.DataFrame, preprocessors):
        # assumes prepare_data logic is invertible
        df_proc, *_ = prepare_data(df, self.cfg)
        return df_proc

    def _export(self, model, study, emb_sizes, preprocessors, metrics):
        self.export_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), self.export_dir / self.cfg.export.weights)
        joblib.dump(preprocessors["num_scaler"], self.export_dir / self.cfg.export.in_scaler)
        joblib.dump(preprocessors["tar_scaler"], self.export_dir / self.cfg.export.tar_scaler)
        joblib.dump(preprocessors["cat_encoder"], self.export_dir / self.cfg.export.cat_encoder)

        with open(self.export_dir / self.cfg.export.metadata, "w") as f:
            json.dump(
                {
                    "best_params": study.best_params,
                    "emb_sizes": emb_sizes,
                    "num_cols": self.cfg.data.num_cols,
                    "cat_cols": self.cfg.data.cat_cols,
                    "target_names": self.cfg.data.target_cols,
                    "metrics": metrics,
                },
                f,
                indent=4,
            )


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