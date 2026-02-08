


# src/my_ml_project/registry/mlflow.py

import mlflow
import mlflow.pytorch
from my_ml_project.registry.base import ModelRegistry

class MLflowModelRegistry(ModelRegistry):

    def save(self, model, name: str, version: str | None = None):
        with mlflow.start_run():
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name=name,
            )

    def load(self, name: str, stage: str):
        model_uri = f"models:/{name}/{stage}"
        return mlflow.pytorch.load_model(model_uri)


# Training Pipeline (unchanged logic)
# Model is now registered
# Automatically becomes latest DEV version
# No stage yet



# # src/my_ml_project/registry/mlflow.py

# import mlflow
# import mlflow.pytorch

# from my_ml_project.registry.base import ModelRegistry

# class MLflowModelRegistry(ModelRegistry):

#     def save(self, model, name: str, version: str):
#         with mlflow.start_run(run_name=f"{name}-{version}"):
#             mlflow.pytorch.log_model(
#                 model,
#                 artifact_path="model"
#                 )
#             mlflow.set_tag("model_name", name)
#             mlflow.set_tag("version", version)

#     def load(self, name: str, version: str):
#         model_uri = f"models:/{name}/{version}"
#         return mlflow.pytorch.load_model(model_uri)
