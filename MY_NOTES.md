
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