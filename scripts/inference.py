


pipeline = InferencePipeline(cfg, artifact_dir)

pipeline.predict_batch(df)     # pandas → pandas
pipeline.predict_one(record)   # dict → dict / scalar

def predict_one(self, record: dict) -> dict:
    df = pd.DataFrame([record])
    return self.predict_batch(df).iloc[0].to_dict()


# Batch job
pipeline = InferencePipeline(cfg, artifact_dir)
preds = pipeline.predict_batch(df)
preds.to_parquet("predictions.parquet")


# FastAPI / Flask / BentoML
pipeline = InferencePipeline(cfg, artifact_dir)

@app.post("/predict")
def predict(payload: dict):
    return pipeline.predict_one(payload)


# # File: scripts/inference.py
# # Placeholder for inference script. Expand as needed for loading model, preprocess, predict.

# import torch
# import pandas as pd
# import joblib
# import json
# from pathlib import Path
# from find_project_root import find_project_root  # Adjust
# from demo_ml_package.models.model import DynamicModel
# from demo_ml_package.configs.schema import ConfigSchema
# # Add imports for scalers, encoders, etc.

# def main():
#     PROJECT_ROOT = find_project_root(dirname="moffitt")
    
#     # Load config (subset or full)
#     with open(PROJECT_ROOT / "config" / "config.yaml", "r") as f:
#         cfg_dict = yaml.safe_load(f)
#     cfg = ConfigSchema(**cfg_dict)  # May need only parts

#     export_path = PROJECT_ROOT / cfg.export.dir
    
#     # Load metadata
#     with open(export_path / cfg.export.metadata, 'r') as f:
#         metadata = json.load(f)
    
#     # Load scalers and encoder
#     num_scaler = joblib.load(export_path / cfg.export.in_scaler)
#     tar_scaler = joblib.load(export_path / cfg.export.tar_scaler)
#     cat_encoder = joblib.load(export_path / cfg.export.cat_encoder)
    
#     # Reconstruct model
#     best_params = metadata['best_params']
#     best_hidden_dims = [best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])]
#     model = DynamicModel(
#         emb_sizes=metadata['emb_sizes'],
#         n_numeric=len(metadata['num_cols']),
#         n_targets=len(metadata['target_names']),
#         hidden_dims=best_hidden_dims,
#         dropout=best_params['dropout']
#     )
#     model.load_state_dict(torch.load(export_path / cfg.export.weights))
#     model.eval()
    
#     # Example inference (add your data)
#     # df_new = pd.read_parquet('new_data.parquet')
#     # Process df_new using cat_encoder, num_scaler
#     # Create dataset/loader
#     # Predict and inverse transform with tar_scaler
#     print("Inference placeholder executed.")

# if __name__ == "__main__":
#     main()
    








# # scripts/run_inference.py

# import hydra
# import torch
# from omegaconf import DictConfig
# from pathlib import Path

# from my_ml_project.pipelines.inference_pipeline import run_inference_pipeline

# @hydra.main(version_base=None, config_path="../configs", config_name="config")
# def main(cfg: DictConfig):
#     predictions = run_inference_pipeline(cfg)

#     output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
#     output_path = output_dir / "predictions.pt"

#     torch.save(predictions, output_path)
#     print(f"Predictions saved to {output_path}")

# if __name__ == "__main__":
#     main()


# # run
# python scripts/run_inference.py inference.model_path=models/model.pt

# Final Flow (Training → Inference)
# TRAIN
# scripts/run_training.py
#   ↓
# pipelines/training_pipeline.py
#   ↓
# models/train.py

# INFERENCE
# scripts/run_inference.py
#   ↓
# pipelines/inference_pipeline.py
#   ↓
# models/predict.py