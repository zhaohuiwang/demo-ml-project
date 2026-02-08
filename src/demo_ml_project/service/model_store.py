

# service/model_store.py

from my_ml_project.registry.factory import get_model_registry

def load_models(cfg):
    registry = get_model_registry(cfg)

    champion = registry.load(
        name=cfg.project_name,
        stage="Production",
    )

    challenger = registry.load(
        name=cfg.project_name,
        stage="Staging",
    )

    return champion, challenger


# Load Both Models
# Role	        MLflow Stage
# Champion	    Production
# Challenger	Staging