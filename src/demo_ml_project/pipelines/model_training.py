
# project-root(demo-ml-project)/src/demo_ml_project/pipelines/model_training.py
import mlflow
import mlflow.pytorch
import mlflow.sklearn

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import joblib
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner

from ..configs.schema import RootConfig
from ..configs.artifacts import PreprocessingArtifacts, TrainingArtifacts
from ..data.processing import prepare_data
from ..data.dataset import InputDataset
from ..models.model import DynamicTabularModel
from ..optimization.objective import objective
from ..utils.early_stopping import EarlyStopping
from ..utils.logging import get_logger
from ..utils.helpers import flatten_dict


class TrainingPipeline:
    def __init__(self, cfg: RootConfig):
        self.cfg = cfg
        self.logger = get_logger(self.__class__.__name__)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self._val_loader: DataLoader | None = None
        self._export_dir: Path | None = None

        # MLflow setup
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("demo-ml-tabular-regression")

    def run(self) -> None:
        """
        Orchestrates the full pipeline with MLflow tracking:
        - Starts MLflow run
        - Logs config
        - Runs core training
        - Evaluates
        - Exports artifacts to disk
        - Logs model + artifacts to MLflow
        - Registers model (if successful)
        """
        self._export_dir = Path(self.cfg.export.dir)
        self._export_dir.mkdir(parents=True, exist_ok=True)

        with mlflow.start_run(run_name=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as mlflow_run:
            # 1. Log full flattened config early
            flat_cfg = flatten_dict(self.cfg.model_dump())
            mlflow.log_params(flat_cfg)
            self.logger.info(f"MLflow run started: {mlflow_run.info.run_id}")

            # 2. Core training (pure logic, no MLflow calls inside)
            artifacts = self.train_only()

            # 3. Evaluate
            metrics = self.evaluate(artifacts.model)
            self.logger.info(f"Final validation metrics: {metrics}")
            mlflow.log_metrics(metrics)

            # 4. Export all artifacts to disk first (creates files)
            self._export(artifacts, metrics)

            # 5. Log the model as proper MLflow PyTorch model (with auto-registration)
            registered_model_name = "TabularMultiTargetRegressor"
            try:
                mlflow.pytorch.log_model(
                    artifacts.model,
                    artifact_path="pytorch_model",
                    registered_model_name=registered_model_name,
                    metadata={
                        "description": "Multi-target regression model for health/economic indicators",
                        "best_cv_loss": artifacts.study.best_value,
                        "framework": "PyTorch",
                        "run_id": mlflow_run.info.run_id,
                    },
                )
                self.logger.info(f"Model registered under: {registered_model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to log/register model: {e}")
                # Continue — don't crash the whole run

            # 6. Log raw weights + other artifacts (non-overlapping paths)
            mlflow.log_artifact(
                self._export_dir / self.cfg.export.weights,
                artifact_path="raw_weights"
            )
            mlflow.log_artifact(
                self._export_dir / self.cfg.export.in_scaler,
                artifact_path="preprocessors"
            )
            mlflow.log_artifact(
                self._export_dir / self.cfg.export.tar_scaler,
                artifact_path="preprocessors"
            )
            mlflow.log_artifact(
                self._export_dir / self.cfg.export.cat_encoder,
                artifact_path="preprocessors"
            )
            mlflow.log_artifact(
                self._export_dir / self.cfg.export.metadata,
                artifact_path="metadata"
            )

            self.logger.info("Pipeline completed. MLflow run logged.")
            self.logger.info(f"View in UI: mlflow ui → http://127.0.0.1:5000/#/experiments/0/runs/{mlflow_run.info.run_id}")

    def train_only(self) -> TrainingArtifacts:
        """
        Pure training logic — no MLflow, no export, no evaluation.
        Returns artifacts ready for export and evaluation.
        """
        # 1. Load and minimal clean
        df = self._load_and_clean_data()
        df_clean = df.copy()

        # 2. Optional small hold-out when CV is enabled
        df_hpo = df_clean
        if self.cfg.training.cv.enabled:
            df_hpo, _ = train_test_split(
                df_clean,
                test_size=0.10,
                random_state=self.cfg.training.random_state,
                shuffle=True,
            )
            self.logger.info(f"CV enabled → HPO on {len(df_hpo)} rows")

        # 3. Compute embedding sizes
        emb_sizes = self._compute_embedding_sizes(df_hpo, self.cfg.data.cat_cols)

        # 4. Hyperparameter optimization
        study = self._run_hyperparameter_optimization(df_hpo, emb_sizes)

        best_hparams = study.best_params
        self.logger.info(f"Best CV mean loss: {study.best_value:.6f}")
        self.logger.info(f"Best hyperparameters: {best_hparams}")

        # 5. Final preprocessing on full data
        final_preprocessing = prepare_data(df_clean, self.cfg, fit=True)
        final_processed_df = final_preprocessing.processed_df

        # 6. Create loaders
        train_loader, val_loader = self._create_dataloaders(final_processed_df)
        self._val_loader = val_loader

        # 7. Create final model
        final_model = self._create_model_from_hparams(best_hparams, emb_sizes)

        # 8. Full training
        self._perform_final_full_training(
            model=final_model,
            train_loader=train_loader,
            val_loader=val_loader,
            hparams=best_hparams,
        )

        return TrainingArtifacts(
            model=final_model,
            study=study,
            emb_sizes=emb_sizes,
            preprocessors={
                "cat_encoder": final_preprocessing.cat_encoder,
                "num_scaler": final_preprocessing.num_scaler,
                "tar_scaler": final_preprocessing.tar_scaler,
            },
        )

    def evaluate(self, model: torch.nn.Module) -> Dict[str, float]:
        if self._val_loader is None:
            raise RuntimeError("Validation loader not available")

        self.logger.info("Running final validation pass...")

        model.eval()
        criterion = torch.nn.MSELoss()
        total_loss = 0.0
        n_samples = 0

        with torch.inference_mode():
            for x_cat, x_num, y_true in self._val_loader:
                x_cat = x_cat.to(self.device, non_blocking=True)
                x_num = x_num.to(self.device, non_blocking=True)
                y_true = y_true.to(self.device, non_blocking=True)

                pred = model(x_cat, x_num)
                loss = criterion(pred, y_true)

                total_loss += loss.item() * len(y_true)
                n_samples += len(y_true)

        if n_samples == 0:
            return {"val_rmse": float("inf"), "val_mse": float("inf")}

        mse = total_loss / n_samples
        return {"val_rmse": mse ** 0.5, "val_mse": mse}

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    def _load_and_clean_data(self) -> pd.DataFrame:
        path = Path(self.cfg.data.train_data_path)
        if not path.exists():
            raise FileNotFoundError(f"Training data not found: {path}")

        df = pd.read_parquet(path, engine="pyarrow")
        if self.cfg.data.drop_columns:
            df = df.drop(columns=self.cfg.data.drop_columns, errors="ignore")

        if "population" in df.columns:
            df["population"] = df["population"].fillna(df["population"].median()).astype("int32")

        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
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
            num_workers=2,
            pin_memory=self.device.type != "cpu",
            persistent_workers=True,
        )

        return (
            DataLoader(train_ds, shuffle=True, **common_kw),
            DataLoader(val_ds, shuffle=False, **common_kw),
        )

    def _run_hyperparameter_optimization(self, df_hpo: pd.DataFrame, emb_sizes):
        pruner_map = {"median": MedianPruner(), "hyperband": HyperbandPruner(), "nop": optuna.pruners.NopPruner()}
        sampler_map = {"tpe": TPESampler(multivariate=True, group=True)}

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler_map.get(self.cfg.optuna.sampler, TPESampler()),
            pruner=pruner_map.get(self.cfg.optuna.pruner, MedianPruner()),
        )

        study.optimize(
            lambda trial: objective(
                trial,
                self.cfg,
                df_hpo,
                emb_sizes,
                self.device
            ),
            n_trials=self.cfg.optuna.n_trials,
            show_progress_bar=True,
        )

        return study

    def _create_model_from_hparams(self, hparams: dict, emb_sizes):
        hidden_dims = [hparams[f"n_units_l{i}"] for i in range(hparams["n_layers"])]
        return DynamicTabularModel(
            emb_sizes=emb_sizes,
            n_numeric=len(self.cfg.data.num_cols),
            n_targets=len(self.cfg.data.target_cols),
            hidden_dims=hidden_dims,
            dropout=hparams["dropout"],
        ).to(self.device)

    def _perform_final_full_training(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        hparams: dict,
    ):
        optimizer_name = hparams.get("optimizer", "Adam")
        lr = hparams["lr"]

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

        criterion = torch.nn.MSELoss()

        early_stopping = EarlyStopping(patience=self.cfg.training.patience)
        best_epoch = 0

        self.logger.info("Starting final full training with selected hyperparameters...")

        for epoch in range(1, self.cfg.training.max_epochs + 1):
            model.train()
            train_loss_total = 0.0
            n_train = 0

            for x_cat, x_num, y in train_loader:
                x_cat = x_cat.to(self.device, non_blocking=True)
                x_num = x_num.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                pred = model(x_cat, x_num)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                train_loss_total += loss.item() * len(y)
                n_train += len(y)

            model.eval()
            val_loss_total = 0.0
            n_val = 0

            with torch.no_grad():
                for x_cat, x_num, y in val_loader:
                    x_cat = x_cat.to(self.device, non_blocking=True)
                    x_num = x_num.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    pred = model(x_cat, x_num)
                    loss = criterion(pred, y)
                    val_loss_total += loss.item() * len(y)
                    n_val += len(y)

            val_loss = val_loss_total / n_val if n_val > 0 else float("inf")
            train_loss = train_loss_total / n_train if n_train > 0 else float("inf")

            self.logger.info(f"Epoch {epoch:3d} | Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(best val loss: {early_stopping.best_loss:.6f})"
                )
                break

            best_epoch = epoch

        # Load best weights if available
        if early_stopping.best_model_state is not None:
            model.load_state_dict(early_stopping.best_model_state)
            self.logger.info(f"Restored best weights (val loss: {early_stopping.best_loss:.6f})")

        self.logger.info(f"Final training completed (best epoch: {best_epoch})")

    def _export(self, artifacts: TrainingArtifacts, metrics: dict):
        if self._export_dir is None:
            raise RuntimeError("Export directory not set")

        p = self._export_dir

        torch.save(artifacts.model.state_dict(), p / self.cfg.export.weights)

        joblib.dump(artifacts.preprocessors["num_scaler"], p / self.cfg.export.in_scaler)
        joblib.dump(artifacts.preprocessors["tar_scaler"], p / self.cfg.export.tar_scaler)
        joblib.dump(artifacts.preprocessors["cat_encoder"], p / self.cfg.export.cat_encoder)

        metadata = {
            "best_hparams": artifacts.study.best_params,
            "best_cv_loss": artifacts.study.best_value,
            "emb_sizes": artifacts.emb_sizes,
            "numeric_features": self.cfg.data.num_cols,
            "categorical_features": self.cfg.data.cat_cols,
            "targets": self.cfg.data.target_cols,
            "final_val_metrics": metrics,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        (p / self.cfg.export.metadata).write_text(json.dumps(metadata, indent=2))