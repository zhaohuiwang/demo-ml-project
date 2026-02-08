# scripts/promote_model.py

import mlflow
from mlflow.tracking import MlflowClient

def promote_to_staging(model_name: str, version: str):
    client = MlflowClient()

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=True,
    )

# def promote_to_prod(model_name: str, version: str):
#     client = MlflowClient()

#     client.transition_model_version_stage(
#         name=model_name,
#         version=version,
#         stage="Production",
#         archive_existing_versions=True,
#     )


if __name__ == "__main__":
    promote_to_staging("my_ml_project", "3")


# Promote DEV → STAGING (Automated)
# This happens via CI / script / job, not training.
# Promotion is never manual vibes.
# It must be based on explicit rules.
# Typical gates:
# metrics threshold (e.g. accuracy >= 0.92 and latency <= 50ms)
# regression test vs prod
# data drift checks
# approval (human or CI)

# Previous staging model archived
# New version now Staging