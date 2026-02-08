

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from my_ml_project.service.model_loader import load_model

app = FastAPI()
model = None

class PredictRequest(BaseModel):
    inputs: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[int]

@app.on_event("startup")
def startup():
    global model
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../../../configs"):
        cfg = compose(config_name="config")

    model = load_model(cfg)

@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
def predict(req: PredictRequest):
    x = torch.tensor(req.inputs, dtype=torch.float32)
    logits = model(x)
    preds = torch.argmax(logits, dim=1)

    return PredictResponse(predictions=preds.tolist())


# Run the API
# uvicorn my_ml_project.service.app:app --reload

# Request:
# curl -X POST http://localhost:8000/predict \
#   -H "Content-Type: application/json" \
#   -d '{"inputs": [[0.1, 0.2, ...]]}'



# FastAPI Canary Inference
# service/app.py (core logic)

router = CanaryRouter(canary_pct=0.05)
champion_model, challenger_model = load_models(cfg)

@app.post("/predict")
def predict(req: PredictRequest):
    route = router.route()

    model = challenger_model if route == "challenger" else champion_model

    preds = infer(model, req.inputs)

    log_prediction(
        route=route,
        inputs=req.inputs,
        preds=preds,
    )

    return {
        "predictions": preds,
        "served_by": route,
    }

# 5% traffic to challenger
# 95% still safe
# Instant rollback (set % to 0)

# Champion - Challenger Rule
# Promote challenger if:
# - accuracy_challenger ≥ accuracy_champion + 1%
# - latency_challenger ≤ latency_champion + 10ms
# - error_rate_challenger < 0.1%


# These rules live in CI or a controller, not the API.

# Rollback Mechanism (Instant)
# # rollback = set canary_pct to 0
# router.canary_pct = 0.0

# Optional: Full Rollback (MLflow)

# If challenger is bad, archive it:

# from mlflow.tracking import MlflowClient

# client.transition_model_version_stage(
#     name="my_ml_project",
#     version="7",
#     stage="Archived",
# )

# Promotion After Canary Success

# Once challenger wins:

# Step A: Promote in MLflow
# client.transition_model_version_stage(
#     name="my_ml_project",
#     version="7",
#     stage="Production",
#     archive_existing_versions=True,
# )

# Step B: Reload Models
# champion_model, challenger_model = load_models(cfg)
# router.canary_pct = 0.0


# Challenger is now champion.

# TRAIN
#   ↓
# REGISTER (DEV)
#   ↓
# PROMOTE → STAGING
#   ↓
# CANARY (5%)
#   ↓
# COMPARE METRICS
#   ↓
# ┌──────────────┬──────────────┐
# │ PROMOTE PROD │ ROLLBACK     │
# └──────────────┴──────────────┘
