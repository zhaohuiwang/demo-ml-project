
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

```bash

# (demo-ml-project) zhaohuiwang@WangFamily:~/dev/demo-ml-project$ python3 scripts/train.py
```
Hydra a popular configuration management framework in Python. It lets you:
1. Keep parameters outside of code (in YAML files)
2. Easily override them from command line
3. Run many experiments with different configs (sweeps)
4. Get automatic logging, output directories, reproducibility
5. Compose configs hierarchically
6. Instantiate objects directly from config (very powerful)

- Python starts, see __main__ module, and `@hydra.main(...)` decorator
Hydra takes over sys.argv, decorator wraps your main function and intercepts CL execution.
- Hydra parses CL arguments: using  key=value expressions to override(Highest priority); using flag expressions to specify, like `--multirum` (sweep/grid/random search mode), `--cfg`, `-`-package`, `hydra-help`, etc.
- Hydra locates the primary config file through two arguments: `config_path= ...` to specify the directory containing your configuration files relative to your Python application. `config_name= ...` to specify the primary configuration file that acts as the entry point for your configurations' configuration hierarchy. for example, `../<config_path>/<config_name>.yaml`.
- Use OmegaCong to lead the base configuration.
- Composition rules: if you use default: list in your `<config_name>.yaml`, Hydra automatically composes many small YAML files into one final config.
- Creates output directory by default, `./outputs/YYYY-MM-DD/HH-MM-SS`.  Automatic versioning of experiment, copy of the effective config, .hydra/ for saving command, and some metadata. This location/pattern are customizable.
- Changes current working directory (cwd) to the newly created output directory, to conveniently saving artifacts but sometime annoying when loading local files. `hydra.utils.get_original_cwd()`, `os.getcwd()` or `Path.cwd()`

Inside the `scripts/train.py`, there are two critical bridge statements between Hydra's configuration system (which uses OmegaConf under the hood) and Pydantic validation system. 
1. `OmegaConf.resolve(hydra_cfg)` Eagerly resolves all interpolations inside the hydra_cfg object in place.
2. `cfg = RootConfig.model_validate(OmegaConf.to_container(hydra_cfg, resolve=True))`

- `OmegaConf.to_container(hydra_cfg, resolve=True)`
Converts the entire OmegaConf structure (DictConfig / ListConfig) into plain Python primitives: nested dict, list, str, int, float, etc. This is necessary because Pydantic cannot directly consume an OmegaConf object — it expects regular Python dicts/lists. The `resolve=True` flag ensures that any remaining unresolved interpolations are resolved during the conversion (as a safety net).

- `RootConfig.model_validate(...)`
Feeds the plain dict into your Pydantic RootConfig model. Runs all of Pydantic's magic: Type coercion (str → Path, str → int, etc.); Field constraints (gt=0, ge=1, etc.); Custom `@field_validator` and `@model_validator` functions (your path resolution logic); Raises clear ValidationError if anything is wrong (wrong type, missing field, invalid range, bad path, etc.)