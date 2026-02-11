# src/demo_ml_project/pipelines/inference_pipeline.py

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import mlflow
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..configs.inference.schema import InferenceConfig
from ..models.model import DynamicTabularModel  # your model class
from ..utils.logging import get_logger


class InferenceDataset(Dataset):
    """Simple dataset for inference (no targets)."""

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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Categorical
        if self.cat_cols and self.cat_encoder:
            cat_data = row[self.cat_cols].values.astype(str)
            cat_encoded = self.cat_encoder.transform([cat_data])[0]
            x_cat = torch.tensor(cat_encoded, dtype=torch.long)
        else:
            x_cat = torch.empty(0, dtype=torch.long)

        # Numerical
        if self.num_cols and self.num_scaler:
            num_data = row[self.num_cols].values.astype(float).reshape(1, -1)
            num_scaled = self.num_scaler.transform(num_data)[0]
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

        self.model: Optional[DynamicTabularModel] = None
        self.num_scaler = None
        self.tar_scaler = None
        self.cat_encoder = None

        # Will be set after loading
        self.emb_sizes: Optional[list[Tuple[int, int]]] = None
        self.input_cat_cols: list[str] = []
        self.input_num_cols: list[str] = []

    def _determine_device(self) -> torch.device:
        if self.cfg.device is not None:
            return torch.device(self.cfg.device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_model_and_preprocessors(self) -> None:
        """Load model and all necessary preprocessors."""
        self.logger.info(f"Loading model from: {self.cfg.load_from}")

        if self.cfg.load_from == "local":
            self._load_from_local()
        elif self.cfg.load_from == "mlflow":
            self._load_from_mlflow()
        else:
            raise ValueError(f"Unsupported load_from value: {self.cfg.load_from}")

        self.logger.info("Model and preprocessors loaded successfully")

    def _load_from_local(self) -> None:
        export_dir = self.cfg.local.export_dir
        if not export_dir.exists():
            raise FileNotFoundError(f"Export directory not found: {export_dir}")

        # Model architecture reconstruction - you need to know emb_sizes, etc.
        # Option A: load from metadata.json if you saved it
        metadata_path = export_dir / "run_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.emb_sizes = metadata.get("emb_sizes", [])
            self.input_cat_cols = metadata.get("categorical_features", [])
            self.input_num_cols = metadata.get("numeric_features", [])
        else:
            self.logger.warning("No metadata found → using config defaults or fallback")
            # Fallback: use values from cfg.data (if you include data group)
            self.input_cat_cols = self.cfg.data.cat_cols if hasattr(self.cfg, "data") else []
            self.input_num_cols = self.cfg.data.num_cols if hasattr(self.cfg, "data") else []

        # Load model
        model_path = export_dir / "final_model_weights.pth"
        self.model = DynamicTabularModel(
            emb_sizes=self.emb_sizes,
            n_numeric=len(self.input_num_cols),
            n_targets=len(self.cfg.data.target_cols) if hasattr(self.cfg, "data") else 3,  # fallback
            hidden_dims=[128, 64],  # ← you may need to save/load actual hparams
            dropout=0.1,
        ).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Preprocessors
        self.num_scaler = joblib.load(export_dir / "num_scaler.pkl")
        self.tar_scaler = joblib.load(export_dir / "target_scaler.pkl")
        self.cat_encoder = joblib.load(export_dir / "cat_encoder.pkl")

    def _load_from_mlflow(self) -> None:
        model_name = self.cfg.mlflow.model_name
        # Preferred: use alias (set during promotion)
        model_uri = f"models:/{model_name}@champion"
        # Alternative: use stage
        # model_uri = f"models:/{model_name}/Production"

        self.logger.info(f"Loading MLflow model: {model_uri}")

        # Load PyTorch model
        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.to(self.device)
        self.model.eval()

        # Load metadata & preprocessors from the same run
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(model_name, "champion")
        run_id = mv.run_id

        # Download artifacts
        artifact_path = "preprocessors"  # adjust if different
        local_dir = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{artifact_path}")

        self.num_scaler = joblib.load(Path(local_dir) / "num_scaler.pkl")
        self.tar_scaler = joblib.load(Path(local_dir) / "target_scaler.pkl")
        self.cat_encoder = joblib.load(Path(local_dir) / "cat_encoder.pkl")

        # Load emb_sizes, feature names from metadata
        metadata_path = Path(local_dir) / "run_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                meta = json.load(f)
            self.emb_sizes = meta.get("emb_sizes")
            self.input_cat_cols = meta.get("categorical_features", [])
            self.input_num_cols = meta.get("numeric_features", [])
        else:
            self.logger.warning("No metadata in MLflow artifacts → using fallback")


    def predict(self) -> pd.DataFrame:
        if self.model is None:
            self.load_model_and_preprocessors()

        # Load input data
        input_path = self.cfg.input.path
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if self.cfg.input.format == "auto":
            if input_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(input_path)
            else:
                df = pd.read_csv(input_path)
        elif self.cfg.input.format == "parquet":
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)

        self.logger.info(f"Loaded {len(df)} rows from {input_path}")

        # Preprocess
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
            num_workers=0,  # increase if needed
            pin_memory=self.device.type != "cpu",
        )

        predictions = []
        self.model.eval()

        with torch.inference_mode():
            for x_cat, x_num in tqdm(loader, desc="Inference"):
                x_cat = x_cat.to(self.device, non_blocking=True)
                x_num = x_num.to(self.device, non_blocking=True)

                preds_scaled = self.model(x_cat, x_num)
                preds = self.tar_scaler.inverse_transform(preds_scaled.cpu().numpy())

                predictions.append(preds)

        # Combine
        pred_array = torch.cat([torch.tensor(p) for p in predictions], dim=0).numpy()
        pred_df = pd.DataFrame(
            pred_array,
            columns=[f"{self.cfg.output.columns_prefix}{c}" for c in self.cfg.target_cols],
            index=df.index if self.cfg.output.include_index else None,
        )

        return pred_df


    def run(self) -> None:
        """Main entry point."""
        try:
            pred_df = self.predict()

            run_dir = Path(HydraConfig.get().runtime.output_dir)
            output_path = run_dir / self.cfg.output.path.name  # keeps your configured filename

            if self.cfg.output.include_index:
                pred_df.to_csv(output_path)
            else:
                pred_df.to_csv(output_path, index=False)

            self.logger.info(f"Predictions saved to: {output_path}")
            self.logger.info(f"Shape of predictions: {pred_df.shape}")

        except Exception as e:
            self.logger.exception("Inference pipeline failed")
            raise

# Optional: if you want to run directly (for testing)
# if __name__ == "__main__":
#     from omegaconf import OmegaConf
#     from hydra import compose, initialize

#     with initialize(version_base=None, config_path="../../conf/inference"):
#         cfg = compose(config_name="default")
#     pipeline = InferencePipeline(InferenceConfig(**OmegaConf.to_container(cfg)))
#     pipeline.run()