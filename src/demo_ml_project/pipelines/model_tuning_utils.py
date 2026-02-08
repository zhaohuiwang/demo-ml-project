
import torch.optim as optim
from torch import nn
from typing import Dict, Any


def sample_model_params(trial: optuna.Trial, cfg) -> dict:

    # Sample architecture
    #choose how many hidden layers the network has
    n_layers: int = trial.suggest_int('n_layers', *cfg.optuna.layer_range)
    # for each layer, choose how many neurons it has
    hidden_dims: list[int] = [
        trial.suggest_categorical(f'n_units_l{i}', cfg.optuna.units_list) for i in range(n_layers)
    ]

    return {
        "n_layers": n_layers,
        "hidden_dims": hidden_dims,
    }


def sample_optimizer_params(trial: optuna.Trial) -> dict:

    # Learning rate
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)   

    # Optimizer
    optimizer_name = trial.suggest_categorical(
        "optimizer", ["Adam", "SGD", "RMSprop"]
        )
    
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)


    dropout: float = trial.suggest_float('dropout', *cfg.optuna.dropout_range)
    # syntax: trial.suggest_float(name, low, high, log=False) # Each trial gets a different value

    return {
        "optimizer": optimizer,
        "lr": lr,
        "dropout": dropout,
    }


def build_optimizer(
    model: nn.Module,
    params: Dict[str, Any],
) -> optim.Optimizer:
    """
    Factory for optimizers so tuning and training stay identical.
    """

    name = params["optimizer"]
    lr = params["lr"]

    if name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)

    if name == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=params.get("momentum", 0.9),
        )

    if name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr)

    raise ValueError(f"Unknown optimizer: {name}")




def objective(
    trial: optuna.Trial,
    cfg: ConfigSchema,
    train_loader: DataLoader,
    val_loader: DataLoader,
    emb_sizes: list[tuple[int, int]],
    device: torch.device,
    task: str = "regression"
) -> float:

    model_params = sample_model_params(trial, cfg)
    opt_params = sample_optimizer_params(trial)

    model = DynamicModel(
        emb_sizes=emb_sizes,
        n_numeric=len(cfg.data.num_cols),
        n_targets=len(cfg.data.target_cols),
        hidden_dims=model_params["hidden_dims"],
        dropout=model_params["dropout"],
    ).to(device)

    optimizer = build_optimizer(model, opt_params)
    criterion = nn.MSELoss()
    criterion = (
        nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()
        )
    # Task type	                Correct loss
    # Multi-class (1 label)     CrossEntropyLoss
    # Multi-label	            BCEWithLogitsLoss
    # Multi-target regression	MSELoss

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        early_stopping=EarlyStopping(patience=3),
    )

    return trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=cfg.optuna.n_epochs_per_trial,
        trial=trial,
    )

