# Demo ML Project – Tabular Multi-Target Regression

A clean, configurable PyTorch-based pipeline for multi-target regression on tabular data.

## Features

- Hydra for powerful and reproducible configuration management
- Optuna hyperparameter optimization (with optional k-fold / time-series cross-validation)
- Strict prevention of data leakage (preprocessing fitted only on training portions)
- Categorical features via learnable embeddings + numerical features
- Early stopping with best-model checkpointing
- Export of model weights, fitted preprocessors, and metadata for easy inference
- Clear separation between short HPO training and full final training
- Multi-target regression support out of the box
- Docker and docker compose
- Auto run train set in watch-and-train.sh, `watchexec`
- MLflow registry over SQLite (`mlflow.db`)


## Project Structure
```
demo-ml-project/
├── conf/                  # Hydra configuration YAML files
│   ├── training_conf.yaml
│   ├── data/
│   ├── training/
│   ├── optuna/
│   └── export/
├── data/
│   └── processed/
├── exported_model
├── model_export/          # automatically created – contains saved models
├── mlflow.db
├── mlflow_tmp_download
│   └── export
│       ├── cat_encoder.joblib
│       ├── input_scaler.joblib
│       ├── metadata.json
│       ├── model_state.pth
│       └── target_scaler.joblib
├── mlruns
│   └── 1
│       ├── feafc2e349944d0db3099a9d428d5244
│       └── models
├── outputs
│   └── 2026-02-11
│       └── 15-50-18
├── scripts/
│   ├── train.py          # main training entry point
│   └── inference.py      # batch predictions
├── src
│   └── demo_ml_project
│       ├── __init__.py
│       ├── cli.py
│       ├── configs
│       │   ├── data
│       │   ├── inference   # Pydantic class for inference
│       │   └── training    # 
│       ├── data
│       ├── features
│       ├── models
│       │   └── model.py    # call order script > pipeline > model
│       ├── optimization
│       ├── pipelines
│       ├── pipelines
│       │   ├── inference_pipeline.py
│       │   └── model_training.py
│       ├── registry
│       ├── service
│       └── utils
├── README.md
├── pyproject.toml
├── pytest.ini
├── uv.lock
├── watch-and-train.sh
└── zw_notes      # Personal study notes
    ├── MY_NOTES.md
    ├── assemble_repo.py
    ├── assemble_repo.sh
    └── helpers.py
```
## Quick Start

### 1. Installation

```bash
# Recommended: use a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
uv sync     # pyproject.toml
uv pip install -r requirements.txt
```
### 2. Training
Run with default settings (single train/val split, no CV):
```Bash
python scripts/train.py
```
Use a pre-defined CV variant from conf/training/cv3.yaml
```Bash
# If you created conf/training/cv3.yaml with the settings you want
python scripts/train.py training=cv3
```
Note: on MacOS Dataloader num_workers=0, persistent_workers=False whereas on Ubuntu Dataloader num_workers=2, persistent_workers=True

When run on feast, feature_repo, first run 
```Python
python3 scripts/preprocess_for_feast.py
``` 
```Bash
cd feature_repo
feast apply

# Materialize (adjust date range to cover your data)
feast materialize-incremental 2020-01-01
# or full refresh if needed:
# feast materialize 2020-01-01 $(date +%Y-%m-%d)

```
Next either respecify the conf/data/default.yaml or apply a overwrite argument
```Bash
python3 scripts/train.py   data.train_data_path=data/processed/processed_for_feast.parquet
```

Time runs for comparison:
```Bash
time python scripts/train.py training.cv.enabled=true training.cv.n_folds=3
time python scripts/train.py
```
Override parameters directly:
```Bash
python scripts/train.py \
  training.cv.enabled=true \
  training.cv.n_folds=3 \
  optuna.n_trials=50 \
  training.batch_size=128
```
### 3. Inference / Predictions
After training, use the exported artifacts:
```Bash
python scripts/inference.py \
  --export-dir model_export/2025xxxx_xxxxxx \
  --input-csv path/to/new_data.csv \
  --output-csv predictions.csv
```
See `scripts/inference.py` for more options (batch size, device, etc.).
Main Technologies

. PyTorch – model & training
. Hydra + OmegaConf – configuration
. Optuna – hyperparameter optimization
. scikit-learn – preprocessing (OrdinalEncoder, StandardScaler)
. pandas – data handling
. Pydantic – strict config validation
. joblib / torch.save – artifact persistence

### How to run the tests
```Bash
# From project root
pytest -vv

# Run only data processing tests
pytest tests/test_data_processing.py

# With coverage
pytest --cov=src/demo_ml_project

# Fast run (skip slow/mark integration)
pytest -m "not integration"
```
### MLflow
Update TrainingPipeline class — add MLflow logging in key places. After the run, MLflow stores everything in a folder (mlruns/)
```Bash
# Re-run training:
python scripts/train.py
# View the UI
mlflow ui
# FastAPI-based Uvicorn ASGI, runs with multiple worker processes for concurrency
# open the http link, default http://localhost:5000

# Solution to ERROR:    [Errno 98] Address already in use
# List processes using port 5000
lsof -i :5000
# or (more concise)
netstat -tuln | grep 5000
# or (if you have ss installed)
ss -tuln | grep 5000

kill -9 12345   # replace 12345 with your PID

# Now retry:
mlflow ui
```
# clean restart MLflow server
```Bash
# Stop everything
pkill -f mlflow
# then restart clean
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
# Then rerun your script
```

### Inference 
Manually alias a champion model in MLflow first.  MLflow UI > Models > Click a version > Asliase: Add > champion > Run inference script
```Bash
python scripts/inference.py num_workers=4 batch_size=512

```

### Docker build & run
```Bash
# Build
DOCKER_BUILDKIT=1 docker build -t demo-ml:latest .

# Train (GPU + volumes)
docker run --rm -it --gpus all \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/model_export:/app/model_export \
  -v $(pwd)/data:/app/data \
  demo-ml:latest
  ```

```Bash
# Build and start everything
docker compose up --build
# Run training in background
docker compose up -d training
# Run inference (override command)
docker compose run --rm training \
  python scripts/inference.py \
    --export-dir /app/model_export/$(ls -t /app/model_export | head -1) \
    --input-csv /app/data/new_data.csv \
    --output-csv /app/predictions.csv
# Stop everything
docker compose down
# View MLflow UIAlways at http://localhost:5000 — even after restart (data persists in ./mlruns volume)
```



### Input / Outputs

```
Training:
 ├── Hydra run dir:  # <project_root>/outputs/YYYY-MM-DD/HH-MM-SS/
 │     └── logs/
 │
 ├── Local export dir # model_export/ Configured via: yaml files inside  conf/
 │     ├── weights
 │     ├── preprocessors
 │     └── metadata
 │  
 └── MLflow     # mlartifacts/, mlruns/, http://127.0.0.1:5000 
       ├── run params
       ├── metrics
       ├── artifacts
       └── model registry

Inference:
 ├── Hydra run dir
 │     ├── predictions.csv
 │     └── logs/
 │
 ├── Reads from:
 │     ├── local export dir
 │     └── OR MLflow registry
 │
 └── MLflow logging during inference

```

### Docker serving
```Bash
# Build a docker image
mlflow models build-docker \
  --model-uri models:/TabularMultiTargetRegressor@champion \
  --name tabular-regressor
# Verify the new image
docker images
```
### Next Steps / Possible Improvements

. X
. Y
. Z

### License
MIT License (or replace with your preferred license)
Happy modeling!
Questions / improvements → feel free to open an issue or PR.


