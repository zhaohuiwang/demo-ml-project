Separate code from data

`src/my_ml_project/` → reusable library code

`data/`, `models/`, `notebooks/` → runtime artifacts

Never import from `data/` or `models/`.

Use` src/` layout (important for packaging)

This prevents accidental imports from the project root.

Pipelines orchestrate, modules do

`models/train.py` → pure ML logic (Rules: no file paths, no CLI parsing, no environment logic, pure Python, easy to test. fit model, return model, unit-testable, reusable from notebooks, pipelines, APIs)

`pipelines/training_pipeline.py` → orchestration (this file connects components, call steps in order: load data, build features, train model, evaluate, return artifacts.  no CLI parsing, no hardcoded paths, read config, returns results)

`scripts/run_training.py` → entry point (impure code: CLI, config, env vars, file paths, saving artifacts.) You run it like: `python3 scripts/run_training.py -data-path data/processed/train.csv --model-out artifacts/models/model_a.pkl` 
Handles I/O, thin wrapper, easy to swap with Airflow, perfect, kuberflow later

How They Call Each Other (Flow)
```bash
TRAIN
scripts/run_training.py
  ↓
pipelines/training_pipeline.py
  ↓
models/train.py

INFERENCE
scripts/run_inference.py
  ↓
pipelines/inference_pipeline.py
  ↓
models/predict.py

```
Dependencies flow inwardm never outward. If you delete `scripts/`, your package should still be importable and testable.

🔒 Training and inference are fully decoupled

🔁 Same model definition, different pipelines

🧪 Easy to test inference with dummy tensors

☁️ Drop-in replacement for batch jobs or APIs

📦 No Hydra leakage into core logic

Your pipeline should call logic, not contain it.
```python
# pipelines/training_pipeline.py
from my_ml_project.data.loaders import load_data
from my_ml_project.models.train import train

def run():
    X, y = load_data()
    model = train(X, y)
    return model
```

pyproject.toml (Minimal Example)
```toml
[project]
name = "my-ml-project"
version = "0.1.0"
description = "ML project for packaging"
requires-python = ">=3.9"
dependencies = [
    "numpy",
    "pandas",
    "scikit-learn"
]

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```
Install locally:
```bash
pip install -e .
```scripts/run_training.py
        ↓
pipelines/training_pipeline.py
        ↓
models/train.py


Optional but Very Useful
CLI entry point
```toml
[project.scripts]
my-ml = "my_ml_project.cli:main"
```
```bash
my-ml train
my-ml predict
```
