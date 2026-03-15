

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import mlflow.pytorch
import pandas as pd
import torch
from demo_ml_project.data.processing import prepare_data
# ... import your config/schema if needed

app = FastAPI(title="Tabular Multi-Target Predictor")

model = mlflow.pytorch.load_model("models:/TabularMultiTargetRegressor/latest")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    # Preprocess (fit=False, use saved preprocessors if you exported them)
    # For simplicity, assume same columns; in production load scalers/encoder
    processed = prepare_data(df, cfg=None, fit=False, ...)  # adapt as needed
    ds = InputDataset(processed.processed_df, ...)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)

    preds = []
    with torch.no_grad():
        for x_cat, x_num, _ in loader:
            out = model(x_cat.to(model.device), x_num.to(model.device))
            preds.append(out.cpu().numpy())

    return {"predictions": np.concatenate(preds).tolist()}