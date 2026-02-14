import os
import json
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

import optuna
import pandas as pd
import torch
from feast import FeatureStore
from hydra.core.hydra_config import HydraConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from ..configs.training.schema import RootConfig
from ..configs.training.artifacts import TrainingArtifacts
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

        # Feast
        self.store = FeatureStore(repo_path=str(cfg.feast.repo_path))
        self.entity_col = cfg.feast.entity_column
        self.ts_col = cfg.feast.event_timestamp_column

        # MLflow
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

        self.val_loader = None
        self.export_dir = None
        self.sample_processed_df = None

        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Feast repo: {cfg.feast.repo_path}")
        self.logger.info(f"MLflow experiment: {cfg.mlflow.experiment_name}")

    def run(self):
        self.export_dir = Path(self.cfg.export.dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        with mlflow.start_run(run_name=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as mlflow_run:
            flat_cfg = flatten_dict(self.cfg.model_dump())
            mlflow.log_params(flat_cfg)

            mlflow.log_param("feast_repo_path", str(self.cfg.feast.repo_path))
            mlflow.log_param("feast_entity_col", self.entity_col)

            artifacts = self.train_only()

            metrics = self.evaluate(artifacts.model)
            self.logger.info(f"Final validation metrics: {metrics}")
            mlflow.log_metrics(metrics)

            self._export(artifacts, metrics)

            self._log_and_register_model(artifacts.model, metrics, mlflow_run.info.run_id)

            mlflow.log_artifacts(str(self.export_dir), "export")

            self.logger.info("Pipeline completed. MLflow run fully logged.")

    def train_only(self) -> TrainingArtifacts:
        entity_df = self._load_entity_df()

        training_service = self.store.get_feature_service("training_features")

        training_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=training_service
        ).to_df()

        self.logger.info(f"Fetched {len(training_df):,} rows from Feast")

        self.sample_processed_df = training_df.sample(n=4, random_state=42)

        df_hpo = training_df
        if self.cfg.training.cv.enabled:
            df_hpo, _ = train_test_split(
                training_df,
                test_size=0.10,
                random_state=self.cfg.training.random_state,
                shuffle=True,
            )

        emb_sizes = self._compute_embedding_sizes(df_hpo, self.cfg.data.cat_cols)

        study = self._run_hyperparameter_optimization(df_hpo, emb_sizes)

        train_loader, val_loader = self._create_dataloaders(training_df)
        self.val_loader = val_loader

        final_model = self._create_model_from_hparams(study.best_params, emb_sizes)

        self._perform_final_full_training(
            model=final_model,
            train_loader=train_loader,
            val_loader=val_loader,
            hparams=study.best_params,
        )

        return TrainingArtifacts(
            model=final_model,
            study=study,
            emb_sizes=emb_sizes,
        )

    def _load_entity_df(self) -> pd.DataFrame:
        path = self.cfg.data.train_data_path
        cols = [self.entity_col]
        if self.ts_col:
            cols.append(self.ts_col)

        df = pd.read_parquet(path, columns=cols)
        if self.ts_col is None:
            df["event_timestamp"] = pd.Timestamp.now()

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
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(multivariate=True, group=True),
            pruner=HyperbandPruner(),
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

        self.logger.info("Starting final full training...")

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
                self.logger.info(f"Early stopping at epoch {epoch} (best val loss: {early_stopping.best_loss:.6f})")
                break

            best_epoch = epoch

        if early_stopping.best_model_state is not None:
            model.load_state_dict(early_stopping.best_model_state)

        self.logger.info(f"Final training completed (best epoch: {best_epoch})")

    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        if self.val_loader is None:
            raise RuntimeError("Validation loader not available")

        self.logger.info("Running final validation pass...")

        model.eval()
        criterion = torch.nn.MSELoss()
        total_loss = 0.0
        n_samples = 0

        with torch.inference_mode():
            for x_cat, x_num, y_true in self.val_loader:
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

    def _export(self, artifacts: TrainingArtifacts, metrics: dict):
        run_dir = Path(HydraConfig.get().runtime.output_dir)
        p = run_dir / self.cfg.export.dir
        p.mkdir(parents=True, exist_ok=True)

        torch.save(artifacts.model.state_dict(), p / self.cfg.export.weights)

        metadata = {
            "created_at": datetime.now().isoformat(),
            "best_hparams": artifacts.study.best_params,
            "best_cv_loss": float(artifacts.study.best_value),
            "embedding_sizes": artifacts.emb_sizes,
            "numeric_features": self.cfg.data.num_cols,
            "categorical_features": self.cfg.data.cat_cols,
            "target_features": self.cfg.data.target_cols,
            "final_val_metrics": {k: float(v) for k, v in metrics.items()},
            "timestamp": pd.Timestamp.now().isoformat(),
            "feast_repo_path": str(self.cfg.feast.repo_path),
            "feast_entity_col": self.entity_col,
        }

        metadata_path = p / self.cfg.export.metadata
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
        self.logger.info(f"Exported metadata → {metadata_path}")

    def _log_and_register_model(self, model, metrics, run_id):
        try:
            if self.sample_processed_df is None:
                sample_cat = torch.randint(0, 100, (4, len(self.cfg.data.cat_cols)), device="cpu")
                sample_num = torch.randn(4, len(self.cfg.data.num_cols), device="cpu")
            else:
                sample_df = self.sample_processed_df.sample(n=4, random_state=42)
                sample_cat = torch.tensor(sample_df[self.cfg.data.cat_cols].values, device="cpu")
                sample_num = torch.tensor(sample_df[self.cfg.data.num_cols].values, device="cpu")

            sample_input = (sample_cat, sample_num)

            with torch.no_grad():
                sample_output = model(sample_cat.to(self.device), sample_num.to(self.device)).cpu()

            signature = infer_signature(sample_input, sample_output)
            input_example = {
                "cat": sample_cat.numpy().tolist(),
                "num": sample_num.numpy().tolist()
            }

            mlflow.pytorch.log_model(
                model,
                "model",
                registered_model_name="TabularMultiTargetRegressor",
                signature=signature,
                input_example=input_example,
                metadata={
                    "description": "Multi-target regression model",
                    "best_cv_loss": artifacts.study.best_value,
                    "run_id": run_id,
                },
            )
            self.logger.info("Model logged and registered to MLflow")

        except Exception as e:
            self.logger.error(f"Failed to log/register model: {e}")