
# project-root(demo-ml-project)/src/demo_ml_project/pipelines/model_training.py
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import joblib
import optuna
import pandas as pd
import torch

import mlflow
import mlflow.pytorch
import mlflow.sklearn 
from mlflow.models.signature import infer_signature

from hydra.core.hydra_config import HydraConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner

from ..configs.training.schema import RootConfig
from ..configs.training.artifacts import TrainingArtifacts
from ..data.processing import prepare_data
from ..data.dataset import InputDataset
from ..models.model import DynamicTabularModel
from ..optimization.objective import objective
from ..utils.early_stopping import EarlyStopping
from ..utils.logging import get_logger
from ..utils.helpers import flatten_dict


### MLflow manual Logging - Complete Control, Custom Workflow   
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
        # Priority: env var > Hydra config > default
        self.tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            self.cfg.mlflow.tracking_uri if hasattr(self.cfg, "mlflow") else "http://127.0.0.1:5000"
        )

        mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(self.cfg.mlflow.experiment_name)

        # store sample for signature
        self.sample_processed_df: pd.DataFrame | None = None
        
        self.logger.info(f"MLflow tracking URI: {self.tracking_uri}")
        self.logger.info(f"MLflow experiment: {self.cfg.mlflow.experiment_name}")

    def run(self) -> None:
        """
        Orchestrates the full pipeline with MLflow tracking:
        - Starts MLflow run
        - Logs config
        - Runs core training
        - Evaluates
        - Exports artifacts to disk (standardized names)
        - Logs model + artifacts to MLflow
        - Registers model (explicitly) + optional alias/stage
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

            # 4. Export all artifacts to disk first (using standardized names)
            self._export(artifacts, metrics)

            # 5. Log the model as proper MLflow PyTorch model
            registered_model_name = self.cfg.mlflow.registered_model_name
            model_artifact_path = self.cfg.mlflow.model_artifact_path

            try:
                # Use the real processed sample as model signature
                if self.sample_processed_df is None:
                    self.logger.warning("No sample processed data available → using random fallback for signature")
                    sample_cat = torch.randint(0, 100, (4, len(self.cfg.data.cat_cols)), dtype=torch.long, device="cpu")
                    sample_num = torch.randn(4, len(self.cfg.data.num_cols), dtype=torch.float32, device="cpu")
                else:
                    sample_df = self.sample_processed_df.sample(n=4, random_state=42)
                    sample_cat = torch.tensor(sample_df[self.cfg.data.cat_cols].values, dtype=torch.long, device="cpu")
                    sample_num = torch.tensor(sample_df[self.cfg.data.num_cols].values, dtype=torch.float32, device="cpu")

                # Generate signature for the torch model
                sample_input = torch.cat([sample_cat, sample_num], dim=1)

                with torch.no_grad():
                    sample_output = artifacts.model(sample_input).cpu()

                sample_input_df = pd.DataFrame(sample_input, columns=self.cfg.data.cat_cols+self.cfg.data.num_cols)
                sample_output_df = pd.DataFrame(sample_output, columns=self.cfg.data.target_cols)


                signature = infer_signature(sample_input_df, sample_output_df)

                mlflow.pytorch.log_model(
                    artifacts.model,
                    name=model_artifact_path,
                    registered_model_name=registered_model_name,
                    signature=signature,
                    metadata={
                        "description": "Multi-target regression model for health/economic indicators",
                        "best_cv_loss": artifacts.study.best_value,
                        "framework": "PyTorch",
                        "run_id": mlflow_run.info.run_id,
                    },
                )
                self.logger.info(f"Model logged and auto-registered under: {registered_model_name}")

                # Get latest version
                from mlflow import MlflowClient
                client = MlflowClient()
                versions = client.search_model_versions(f"name='{registered_model_name}'")
                if versions:
                    latest_version = max(versions, key=lambda v: int(v.version)).version
                    self.logger.info(f"Registered model version: {latest_version}")
                    # Optional: set alias or stage here if desired
                else:
                    self.logger.warning("No versions found after registration")

            except Exception as e:
                self.logger.error(f"Failed to log/register model: {e}", exc_info=True)
                raise

            # 6. Standardized artifact logging – clean and consistent
            export_path = self._export_dir

            # Option A: Log the entire export folder at once (easiest)
            mlflow.log_artifacts(
                local_dir=str(export_path),
                artifact_path="export",           # → appears as /export/model_state.pth etc. in UI
            )
            self.logger.info("All exported files logged to MLflow under 'export/' folder")

            # Option B: Log key files with semantic paths (more discoverable when browsing). A and B don't conflict
            mlflow.log_artifact(export_path / self.cfg.export.weights,        "model")
            mlflow.log_artifact(export_path / self.cfg.export.input_scaler,   "preprocessors")
            mlflow.log_artifact(export_path / self.cfg.export.target_scaler,  "preprocessors")
            if (export_path / self.cfg.export.cat_encoder).exists():
                mlflow.log_artifact(export_path / self.cfg.export.cat_encoder, "preprocessors")
            mlflow.log_artifact(export_path / self.cfg.export.metadata,       "preprocessors")

            self.logger.info("Pipeline completed. MLflow run fully logged.")
            self.logger.info(
                f"View in UI: mlflow ui → http://127.0.0.1:5000/#/experiments/"
                f"{mlflow_run.info.experiment_id}/runs/{mlflow_run.info.run_id}"
            )
            self.logger.info(
                f"Model registered as: models:/{registered_model_name}/latest "
                f"(use @champion alias after promotion if configured)"
            )

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

        # Save small sample for signature
        self.sample_processed_df = final_processed_df.sample(n=4, random_state=42)
        self.logger.debug(f"Saved sample of {len(self.sample_processed_df)} rows for model signature")

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
                x_all = torch.cat([x_cat, x_num], dim=1).to(self.device, non_blocking=True)
                y_true = y_true.to(self.device, non_blocking=True)

                pred = model(x_all)
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
                x_all = torch.cat([x_cat, x_num], dim=1).to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                pred = model(x_all)
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
                    x_all = torch.cat([x_cat, x_num], dim=1).to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    pred = model(x_all)
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
        # Save to Hydra's per-run output dir
        run_dir = Path(HydraConfig.get().runtime.output_dir)
        
        p = run_dir / self._export_dir
        p.mkdir(parents=True, exist_ok=True)

        # ─── Model state ────────────────────────────────────────
        torch.save(artifacts.model.state_dict(), p / self.cfg.export.weights)

        # ─── Preprocessors ──────────────────────────────────────
        joblib.dump(artifacts.preprocessors["num_scaler"],   p / self.cfg.export.input_scaler)
        joblib.dump(artifacts.preprocessors["tar_scaler"],   p / self.cfg.export.target_scaler)
        if "cat_encoder" in artifacts.preprocessors and artifacts.preprocessors["cat_encoder"] is not None:
            joblib.dump(artifacts.preprocessors["cat_encoder"], p / self.cfg.export.cat_encoder)

        # ─── Metadata (single source of truth) ──────────────────
        metadata = {
            "created_at": datetime.now().isoformat(),
            "best_hparams": artifacts.study.best_params,
            "best_cv_loss": float(artifacts.study.best_value),
            "embedding_sizes": artifacts.emb_sizes,               # critical for model reconstruction
            "numeric_features": self.cfg.data.num_cols,
            "categorical_features": self.cfg.data.cat_cols,
            "target_features": self.cfg.data.target_cols,
            "final_val_metrics": {k: float(v) for k, v in metrics.items()},
            "final_val_metrics": metrics,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        metadata_path = p / self.cfg.export.metadata
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))  # handles non-serializable types
        self.logger.info(f"Exported metadata → {metadata_path}")