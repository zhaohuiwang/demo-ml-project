
# Quick retry and test
```bash
python -c "
from pathlib import Path
import yaml
from demo_ml_project.configs.schema import ConfigSchema

root = Path('.').resolve()
with open(root / 'config/config.yaml') as f:
    data = yaml.safe_load(f)
cfg = ConfigSchema(**data, project_root=root)
print('Config OK')
"
````

# Quick test if your project is importable
```bash
# Option A – temporary PYTHONPATH (good for testing)
export PYTHONPATH=$PWD/src:$PYTHONPATH
python -c "from demo_ml_project.configs.schema import RootConfig; print('OK')"

```

```bash
# Basic run (uses all default values from YAML files)
python scripts/train.py

# Most common useful overrides – examples
python scripts/train.py training.batch_size=128 training.max_epochs=50

python scripts/train.py optuna.n_trials=30 optuna.n_epochs_per_trial=10

python scripts/train.py device=cpu

python scripts/train.py data.train_data_path=data/processed/my_other_dataset.parquet

# Combine several overrides
python scripts/train.py \
    training.batch_size=256 \
    optuna.n_trials=50 \
    optuna.dropout_range="[0.05,0.45]" \
    device=mps

# Force a specific output dir
python scripts/train.py hydra.run.dir=outputs/my-experiment-001

# Try 4 combinations in parallel (or sequentially if no --multirun launcher)
python scripts/train.py -m \
    training.batch_size=64,128,256 \
    optuna.n_trials=10
```

# Hydra automatically creates timestamped folders:
```text
outputs/
├── 2026-02-08/
│   ├── 15-42-33/               ← each run gets its own folder
│   │   ├── .hydra/             ← very useful for reproducibility!
│   │   │   ├── config.yaml
│   │   │   ├── hydra.yaml
│   │   │   └── overrides.yaml
│   │   ├── logs/
│   │   │   └── train_2026-02-08-15-42-33.log
│   │   └── model_export/       ← your export.dir
│   └── ...
```


# look for files and lines that have the specified string pattern  ("Literal.*cuda\|Literal.*device")in the specified dir (src/)
```bash
grep -r -i "Literal.*cuda\|Literal.*device" src/
# find any file and lines with "structured" inside the src/
grep -r "structured" src/
```

# After the change. Clean pycache (optional but safe):
```bash
find src -type d -name __pycache__ -exec rm -rf {} +
```

