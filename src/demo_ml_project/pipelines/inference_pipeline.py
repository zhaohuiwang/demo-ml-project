# src/demo_ml_project/pipelines/inference_pipeline.py

import json
from pathlib import Path
from typing import Optional

import joblib
import mlflow

import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from demo_ml_project.configs.inference.schema import InferenceConfig
from demo_ml_project.models.model import DynamicTabularModel
from demo_ml_project.utils.logging import get_logger


class InferenceDataset(Dataset):
    """Dataset for inference (features only, no targets)."""

    def __init__(
        self,
        df: pd.DataFrame,
        cat_cols: list[str],
        num_cols: list[str],
        cat_encoder=None,
        num_scaler=None,
    ):
        self.df = df
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.cat_encoder = cat_encoder
        self.num_scaler = num_scaler

        # Safety check: required columns must exist
        missing_cat = [c for c in cat_cols if c not in df.columns]
        missing_num = [c for c in num_cols if c not in df.columns]
        if missing_cat or missing_num:
            raise ValueError(
                f"Input DataFrame is missing required columns:\n"
                f"  Missing categorical: {missing_cat}\n"
                f"  Missing numerical:   {missing_num}"
            )

        # Optional: log column order for debugging
        if cat_cols:
            print(f"Using categorical columns (order matters): {cat_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Categorical — wrap as DataFrame with exact column names
        if self.cat_cols and self.cat_encoder:
            # Extract values → force correct column names & order
            cat_values = row[self.cat_cols].values.astype(str).reshape(1, -1)
            # Created a DataFrame to avoid scikit-learn warning
            cat_df = pd.DataFrame(cat_values, columns=self.cat_cols)
            # Now transform with named DataFrame, guaranteed order
            cat_encoded = self.cat_encoder.transform(cat_df)[0]
            x_cat = torch.tensor(cat_encoded, dtype=torch.long)
        else:
            x_cat = torch.empty(0, dtype=torch.long)

        # Now work on Numerical
        if self.num_cols and self.num_scaler:
            num_values = row[self.num_cols].values.astype(float).reshape(1, -1)
            num_df = pd.DataFrame(num_values, columns=self.num_cols)   
            num_scaled = self.num_scaler.transform(num_df)[0]
            x_num = torch.tensor(num_scaled, dtype=torch.float32)
        else:
            x_num = torch.empty(0, dtype=torch.float32)

        return x_cat, x_num


class InferencePipeline:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.logger = get_logger(self.__class__.__name__)

        self.device = self._determine_device()
        self.logger.info(f"Using device: {self.device}")

        # Loaded components
        self.model: Optional[DynamicTabularModel] = None
        self.num_scaler = None
        self.tar_scaler = None
        self.cat_encoder = None

        # Metadata-derived attributes
        self.emb_sizes: Optional[list] = None
        self.input_cat_cols: list[str] = []
        self.input_num_cols: list[str] = []
        self.target_features: list[str] = []   # ← key for output column names

    def _determine_device(self) -> torch.device:
        if self.cfg.device is not None:
            return torch.device(self.cfg.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_model_and_preprocessors(self) -> None:
        """Central entry point to load everything needed for inference."""
        self.logger.info(f"Loading model from source: {self.cfg.load_from}")
        
        if self.cfg.load_from == "local":
            self._load_from_local()
        elif self.cfg.load_from == "mlflow":
            self._load_from_mlflow()
        else:
            raise ValueError(f"Unsupported load_from value: {self.cfg.load_from}")

        self.logger.info("Model and all preprocessors loaded successfully")

    def _load_from_local(self) -> None:
        p = Path(self.cfg.local.export_dir).resolve()
        self.logger.info(f"Loading from local directory: {p}")

        metadata_path = p / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"metadata.json is required: {metadata_path}")

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        self.emb_sizes = meta.get("embedding_sizes")
        if self.emb_sizes is None:
            raise ValueError("metadata.json missing required key: 'embedding_sizes'")

        self.input_cat_cols = meta.get("categorical_features", [])
        self.input_num_cols = meta.get("numeric_features", [])
        self.target_features = meta.get("target_features", self.cfg.target_cols or ["target"])

        hparams = meta.get("best_hparams", {})
        n_layers = hparams.get("n_layers", 2)
        hidden_dims = [hparams.get(f"n_units_l{i}", 128 // (2**i)) for i in range(n_layers)]

        self.model = DynamicTabularModel(
            emb_sizes=self.emb_sizes,
            n_numeric=len(self.input_num_cols),
            n_targets=len(self.target_features),
            hidden_dims=hidden_dims,
            dropout=hparams.get("dropout", 0.1),
        ).to(self.device)

        weights_path = p / "model_state.pth"
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.num_scaler  = joblib.load(p / "input_scaler.joblib")
        self.tar_scaler  = joblib.load(p / "target_scaler.joblib")
        if self.input_cat_cols:
            self.cat_encoder = joblib.load(p / "cat_encoder.joblib")

        self.logger.info(f"Local load complete | targets: {self.target_features}")
        

    def _load_from_mlflow(self) -> None:
        if not hasattr(self.cfg, "mlflow") or self.cfg.mlflow is None:
            raise ValueError("MLflow configuration missing in inference config")

        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        model_name = self.cfg.mlflow.model_name
        alias = getattr(self.cfg.mlflow, "alias", "champion")

        if not model_name:
            raise ValueError("mlflow.model_name is required when load_from = 'mlflow'")

        model_uri = f"models:/{model_name}@{alias}"
        self.logger.info(f"Loading MLflow model: {model_uri}")

        # Try fast path first
        try:
            self.model = mlflow.pytorch.load_model(model_uri)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info("Model loaded directly via mlflow.pytorch.load_model")
        except Exception as e:
            self.logger.warning(f"Direct model load failed: {e}. Falling back to artifacts.")

        # Download artifacts
        client = MlflowClient()
        mv = client.get_model_version_by_alias(model_name, alias)
        run_id = mv.run_id

        downloaded_dir = mlflow.artifacts.download_artifacts(
            f"runs:/{run_id}/export",
            dst_path="./mlflow_tmp_download"
        )
        artifact_root = Path(downloaded_dir).resolve()

        # Metadata
        meta_path = artifact_root / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"metadata.json not found in MLflow export: {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.emb_sizes = meta.get("embedding_sizes")
        if self.emb_sizes is None:
            raise ValueError("metadata.json missing 'embedding_sizes'")

        self.input_cat_cols = meta.get("categorical_features", [])
        self.input_num_cols = meta.get("numeric_features", [])
        self.target_features = meta.get("target_features", self.cfg.target_cols or ["target"])

        # Reconstruct model if direct load failed
        if self.model is None:
            hparams = meta.get("best_hparams", {})
            n_layers = hparams.get("n_layers", 2)
            hidden_dims = [hparams.get(f"n_units_l{i}", 128 // (2**i)) for i in range(n_layers)]

            self.model = DynamicTabularModel(
                emb_sizes=self.emb_sizes,
                n_numeric=len(self.input_num_cols),
                n_targets=len(self.target_features),
                hidden_dims=hidden_dims,
                dropout=hparams.get("dropout", 0.1),
            ).to(self.device)

            state_path = artifact_root / "model_state.pth"
            state_dict = torch.load(state_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()

        # Preprocessors
        self.num_scaler  = joblib.load(artifact_root / "input_scaler.joblib")
        self.tar_scaler  = joblib.load(artifact_root / "target_scaler.joblib")
        if self.input_cat_cols:
            self.cat_encoder = joblib.load(artifact_root / "cat_encoder.joblib")

        self.logger.info(f"MLflow load complete | targets: {self.target_features}")

    def predict(self) -> pd.DataFrame:
        if self.model is None:
            self.load_model_and_preprocessors()

        input_path = Path(self.cfg.input.path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        if self.cfg.input.format == "auto":
            suffix = input_path.suffix.lower()
            if suffix == ".parquet":
                df = pd.read_parquet(input_path)
            else:
                df = pd.read_csv(input_path)
        elif self.cfg.input.format == "parquet":
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)

        self.logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        
        # check: verify required columns exist 
        required_cols = self.input_cat_cols + self.input_num_cols

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input data is missing {len(missing_cols)} required columns:\n"
                f"  Missing: {missing_cols}\n"
                f"  Expected (from training): {required_cols}"
            )

        extra_cols = [col for col in df.columns if col not in required_cols]
        if extra_cols:
            self.logger.info(f"Input has {len(extra_cols)} extra columns (will be ignored): {extra_cols[:10]}{'...' if len(extra_cols)>10 else ''}")

        # Optional: enforce exact column order (very safe)
        df = df[required_cols + [c for c in df.columns if c not in required_cols]]

        self.logger.info(f"Input columns verified and ordered: {df.columns.tolist()[:10]}{'...' if len(df.columns)>10 else ''}")
        dataset = InferenceDataset(
            df=df,
            cat_cols=self.input_cat_cols,
            num_cols=self.input_num_cols,
            cat_encoder=self.cat_encoder,
            num_scaler=self.num_scaler,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=getattr(self.cfg, "num_workers", 0),
            pin_memory=self.device.type != "cpu",
        )

        predictions = []
        
        if self.device:
            self.model.to(self.device)
        self.model.eval()

        self.logger.info("Running inference...")
        with torch.inference_mode():
            for x_cat, x_num in tqdm(loader, desc="Predicting", disable=len(loader) < 10):
                x = torch.cat([x_cat, x_num], dim=1).to(self.device, non_blocking=True)

                preds_scaled = self.model(x).cpu().numpy()
                preds = self.tar_scaler.inverse_transform(preds_scaled)
                predictions.append(preds)

        pred_array = np.vstack(predictions)
        self.logger.info(f"Prediction shape: {pred_array.shape}")

        # Output column names: prefer metadata → config → fallback
        target_cols = self.target_features or getattr(self.cfg, "target_cols", None) or [f"target_{i}" for i in range(pred_array.shape[1])]

        prefix = self.cfg.output.columns_prefix or ""
        output_columns = [f"{prefix}{col}" for col in target_cols]

        pred_df = pd.DataFrame(
            pred_array,
            columns=output_columns,
            index=df.index if self.cfg.output.include_index else None,
        )

        return pred_df

    def run(self) -> None:
        try:
            pred_df = self.predict()

            # Save to custom dir
            # run_dir = Path(HydraConfig.get().runtime.output_dir)
            # output_path = run_dir / Path(self.cfg.output.path).name
            # output_path.parent.mkdir(parents=True, exist_ok=True)

            # if output_path.suffix.lower() in {".parquet"}:
            #     pred_df.to_parquet(output_path, index=self.cfg.output.include_index)
            # else:
            #     pred_df.to_csv(output_path, index=self.cfg.output.include_index)

            # self.logger.info(f"Predictions saved → {output_path}")
            # self.logger.info(f"Shape: {pred_df.shape}")
            # self.logger.info(f"Columns: {list(pred_df.columns)}")

            #  Save to Hydra output dir
            run_dir = Path(HydraConfig.get().runtime.output_dir)
            output_path = run_dir / self.cfg.output.path.name
            pred_df.to_csv(output_path, index=self.cfg.output.include_index)
            self.logger.info(f"Predictions saved to: {output_path}")

        except Exception as e:
            self.logger.exception("Inference pipeline failed")
            raise