# project-root(demo-ml-project)/scripts/train.py

from pathlib import Path
from datetime import datetime
import yaml
import logging

from demo_ml_project.configs.schema import ConfigSchema
from demo_ml_project.pipelines.model_training import TrainingPipeline
from demo_ml_project.utils.logging import configure_logging


def main() -> None:
    project_root: Path = Path(__file__).resolve().parents[1]

    logger = logging.getLogger(__name__)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    configure_logging(
        log_dir=project_root / "logs/train",
        log_file=f"train_{timestamp}.log",
        default_level="INFO",
        json_format=False,
    )

    config_path = project_root / "config" / "config.yaml"
    if not config_path.is_file():
        logger.error("Config file not found: %s", config_path)
        raise FileNotFoundError(config_path)

    try:
        with config_path.open(encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        logger.error("Invalid YAML in config file", exc_info=True)
        raise

    try:
        cfg = ConfigSchema.model_validate({**cfg_dict, "project_root": project_root})
    except Exception as e:
        logger.error("Configuration validation failed", exc_info=True)
        raise

    # or inject project_root in the loader function

    # SUGGESTION: consider using context manager or dependency injection in the future
    pipeline = TrainingPipeline(cfg)

    # Full pipeline
    try:
        pipeline.run()
    except Exception:
        logger.exception("Training pipeline failed")
        raise

    # # Training only (no export, no eval)
    # artifacts = TrainingPipeline(cfg, export_dir).train_only()

    # # 
    # pipeline = TrainingPipeline(cfg, export_dir)
    # artifacts = pipeline.train_only()
    # metrics = pipeline.evaluate(artifacts["model"])



if __name__ == "__main__":
    main()







# # File: scripts/train.py
# import pandas as pd
# import torch
# import yaml
# import joblib
# import json
# import optuna
# from datetime import datetime
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader
# from pathlib import Path
# from typing import Any

# # Assuming find_project_root is available; could be imported or copied.
# from find_project_root import find_project_root  # Adjust path if needed

# from demo_ml_package.configs.schema import ConfigSchema
# from demo_ml_package.utils.logging import configure_logging, get_logger
# from demo_ml_package.data.processing import prepare_data
# from demo_ml_package.data.dataset import InputDataset
# from demo_ml_package.models.model import DynamicModel
# from demo_ml_package.optimization.objective import objective

# def main() -> None:
#     PROJECT_ROOT: Path = find_project_root(dirname="moffitt")  # Adjust dirname as needed
#     current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

#     configure_logging(
#         log_dir=PROJECT_ROOT / "logs/train",
#         log_file=f"optuna_{current_datetime}.log",
#         default_level="INFO",
#         json_format=False,
#         )
#     logger = get_logger(__name__)
#     logger.info(f"Project root: {PROJECT_ROOT}")

#     # Load and Validate Config
#     with open(PROJECT_ROOT / "config" / "config.yaml", "r") as f:
#         cfg_dict: dict[str, Any] = yaml.safe_load(f)

#     cfg: ConfigSchema = ConfigSchema(**cfg_dict)

#     # update to get the absolute date path
#     cfg.data.path = PROJECT_ROOT / cfg.data.path

#     export_path: Path = PROJECT_ROOT / cfg.export.dir
#     export_path.mkdir(parents=True, exist_ok=True)
#     logger.info(f"Export path is set as: {export_path}/")

#     # 1. Prepare Data
#     data_path: Path = cfg.data.path
#     data_drop_columns: list[str] = cfg.data.drop_columns

#     df: pd.DataFrame = pd.read_parquet(data_path).drop(columns=data_drop_columns)

#     assert not df.empty, "Dataframe is empty after dropping columns"

#     df['population'] = df['population'].fillna(df['population'].median()).astype('int')
#     df = df.fillna(df.median(numeric_only=True))

#     df_proc, cat_encoder, num_scaler, tar_scaler = prepare_data(df, cfg)

#     # 2. Define Embeddings
#     emb_sizes: list[tuple[int, int]] = [
#         (int(df_proc[col].nunique()), min(50, (int(df_proc[col].nunique()) + 1) // 2)) 
#         for col in cfg.data.cat_cols
#     ]
#     assert len(emb_sizes) == len(cfg.data.cat_cols), "Embedding sizes mismatch with categorical columns"

#     # 3. Setup DataLoaders
#     train_df, val_df = train_test_split(df_proc, test_size=cfg.training.test_size, random_state=cfg.training.random_state)

#     train_loader = DataLoader(
#         InputDataset(train_df, cfg.data.cat_cols, cfg.data.num_cols, cfg.data.target_cols),
#         batch_size=cfg.training.batch_size, shuffle=True
#     )

#     val_loader: DataLoader = DataLoader(
#         InputDataset(val_df, cfg.data.cat_cols, cfg.data.num_cols, cfg.data.target_cols), 
#         batch_size=cfg.training.batch_size
#     )

#     # 4. Run Optuna Optimization
#     device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     study: optuna.Study = optuna.create_study(direction='minimize')

#     try:
#         study.optimize(
#             lambda t: objective(t, cfg, train_loader, val_loader, emb_sizes, device),
#             n_trials=cfg.optuna.n_trials
#         )
#     except Exception as e:
#         logger.exception("Optuna optimization failed")
#         raise RuntimeError("Hyperparameter optimization failed") from e

#     logger.info(f"Optimization finished. Best Params: {study.best_params}")

#     # 5. Initialize Champion Model with Best Params
#     logger.info("Training final Champion Model with best hyperparameters...")
#     best_params = study.best_params
#     best_hidden_dims: list[int] = [best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])]
    
#     champion_model: DynamicModel = DynamicModel(
#         emb_sizes=emb_sizes,
#         n_numeric=len(cfg.data.num_cols),
#         n_targets=len(cfg.data.target_cols),
#         hidden_dims=best_hidden_dims,
#         dropout=best_params['dropout']
#     ).to(device)

#     # 6. Final Export
#     # Note: For a true production run, you might want to retrain for a few epochs here

#     # 6.1. Save Torch Weights
#     torch.save(champion_model.state_dict(), export_path / cfg.export.weights)
    
#     # 6.2. Save Sklearn Objects (Scalers & Encoder)
#     joblib.dump(num_scaler, export_path / cfg.export.in_scaler)
#     joblib.dump(tar_scaler, export_path / cfg.export.tar_scaler)
#     joblib.dump(cat_encoder, export_path / cfg.export.cat_encoder)

#     # 6.3. Save Metadata (To reconstruct architecture during inference)
#     metadata: dict[str, Any] = {
#         "best_params": best_params,
#         "emb_sizes": emb_sizes,
#         "num_cols": cfg.data.num_cols,
#         "cat_cols": cfg.data.cat_cols,
#         "target_names": cfg.data.target_cols
#     }
    
#     with open(export_path / cfg.export.metadata, 'w') as f:
#         json.dump(metadata, f, indent=4)
        
#     logger.info(f"Successfully exported all artifacts to {export_path}/")

# if __name__ == "__main__":
#     main()

    

# # # scripts/run_training.py

# # import hydra
# # import torch
# # from omegaconf import DictConfig
# # from pathlib import Path

# # from my_ml_project.pipelines.training_pipeline import run_training_pipeline

# # @hydra.main(version_base=None, config_path="../configs", config_name="config")
# # def main(cfg: DictConfig):
# #     print(cfg)

# #     model = run_training_pipeline(cfg)

# #     # Save model (Hydra changes cwd!)
# #     output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
# #     model_path = output_dir / "model.pt"

# #     torch.save(model.state_dict(), model_path)
# #     print(f"Model saved to {model_path}")

# # if __name__ == "__main__":
# #     main()



# # # Run it
# # python scripts/run_training.py
# # # Override configs
# # python scripts/run_training.py training.epochs=50 training.lr=1e-4
# # # Change model
# # python scripts/run_training.py model.hidden_dim=512


# # prepare data
# # build loaders
# # build embeddings

# # tuning_pipeline = TuningPipeline(
# #     cfg, train_loader, val_loader, emb_sizes, device
# # )

# # study = tuning_pipeline.run()

# # training_pipeline = TrainingPipeline(cfg, emb_sizes, device)
# # champion_model = training_pipeline.run(
# #     train_loader, val_loader, study.best_params
# # )

# # export_artifacts(...)
# # How They Call Each Other (Flow)
# # scripts/run_training.py
# #         ↓
# # pipelines/training_pipeline.py
# #         ↓
# # models/train.py

# # Testing Strategy 
# # models/train.py → fast unit tests
# # pipelines/training_pipeline.py → integration tests (mock data)
# # scripts/run_training.py → usually not tested (or smoke test only)