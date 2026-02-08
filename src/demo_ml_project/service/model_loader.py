

import torch
from my_ml_project.pipelines.training_pipeline import MLP
from my_ml_project.registry.factory import get_model_registry

def load_model(cfg):
    registry = get_model_registry(cfg)

    model = MLP(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        output_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout,
    )

    state_dict = registry.load(
        name=cfg.project_name,
        version=cfg.inference.version
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model
