


from __future__ import annotations

import json
import joblib
import torch
import pandas as pd

from pathlib import Path
from typing import Dict, Any

from torch.utils.data import DataLoader

from demo_ml_package.configs.schema import ConfigSchema
from demo_ml_package.data.dataset import InputDataset
from demo_ml_package.data.processing import prepare_data
from demo_ml_package.models.model import DynamicModel
from demo_ml_package.utils.logging import get_logger


class InferencePipeline:
    """
    Inference-only pipeline.
    Loads artifacts once, supports batch and realtime prediction.
    """

    def __init__(self, cfg: ConfigSchema, artifact_dir: Path):
        self.cfg = cfg
        self.artifact_dir = artifact_dir
        self.logger = get_logger(self.__class__.__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._load_model()
        self.preprocessors = self._load_preprocessors()

    # ======================================================
    # Public API
    # ======================================================

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch inference (offline, large datasets).
        """
        self.logger.info(f"Running batch inference on {len(df)} rows")

        df_proc = self._preprocess(df)

        dataset = InputDataset(
            df_proc,
            self.cfg.data.cat_cols,
            self.cfg.data.num_cols,
            target_cols=[],  # no targets in inference
        )

        loader = DataLoader(dataset, batch_size=self.cfg.inference.batch_size)
        preds = []

        self.model.eval()
        with torch.no_grad():
            for x_cat, x_num in loader:
                x_cat = x_cat.to(self.device)
                x_num = x_num.to(self.device)
                preds.append(self.model(x_cat, x_num).cpu())

        preds = torch.cat(preds).numpy()
        return pd.DataFrame(preds, columns=self.cfg.data.target_cols)

    def predict_one(self, record: Dict[str, Any]) -> Dict[str, float]:
        """
        Realtime / online inference (single record).
        """
        self.logger.debug("Running realtime inference")

        df = pd.DataFrame([record])
        preds = self.predict_batch(df)

        return preds.iloc[0].to_dict()

    # ======================================================
    # Internal helpers
    # ======================================================

    def _load_model(self) -> DynamicModel:
        self.logger.info("Loading model artifacts")

        with open(self.artifact_dir / self.cfg.export.metadata) as f:
            meta = json.load(f)

        model = DynamicModel(
            emb_sizes=meta["emb_sizes"],
            n_numeric=len(meta["num_cols"]),
            n_targets=len(meta["target_names"]),
            hidden_dims=[
                meta["best_params"][f"n_units_l{i}"]
                for i in range(meta["best_params"]["n_layers"])
            ],
            dropout=meta["best_params"]["dropout"],
        ).to(self.device)

        model.load_state_dict(
            torch.load(self.artifact_dir / self.cfg.export.weights, map_location=self.device)
        )

        return model

    def _load_preprocessors(self):
        return {
            "num_scaler": joblib.load(self.artifact_dir / self.cfg.export.in_scaler),
            "tar_scaler": joblib.load(self.artifact_dir / self.cfg.export.tar_scaler),
            "cat_encoder": joblib.load(self.artifact_dir / self.cfg.export.cat_encoder),
        }

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing consistently with training.
        """
        df_proc, *_ = prepare_data(df, self.cfg)
        return df_proc




# # src/my_ml_project/pipelines/inference_pipeline.py

# import torch
# from my_ml_project.pipelines.basepipeline import BasePipeline
# from my_ml_project.data.loaders import load_inference_dataset
# from my_ml_project.pipelines.training_pipeline import MLP
# from my_ml_project.models.predict import predict


# class InferencePipeline(BasePipeline):
#     def run_inference_pipeline(cfg):
#         device = cfg.inference.device

#         # Dataset
#         dataset = load_inference_dataset(cfg.data.inference_path)

#         # Model
#         model = MLP(
#             input_dim=cfg.model.input_dim,
#             hidden_dim=cfg.model.hidden_dim,
#             output_dim=cfg.model.output_dim,
#             dropout=cfg.model.dropout,
#         )

#         state_dict = torch.load(cfg.inference.model_path, map_location=device)
#         model.load_state_dict(state_dict)

#         # model = registry.load(
#         #     name=cfg.project_name,
#         #     stage="Staging",
#         # )


#         # Predict
#         predictions = predict(
#             model=model,
#             dataset=dataset,
#             batch_size=cfg.inference.batch_size,
#             device=device,
#         )

#         return predictions
    
#     def run(self) -> None:
#         pass

    

# Shares model definition
# Clean separation from scripts
# Works for batch or online inference