
# project-root(demo-ml-project)/src/demo_ml_project/optimization/objective.py

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ..configs.schema import RootConfig
from ..utils.early_stopping import EarlyStopping
from ..models.model import DynamicModel

def objective(
        # String-based annotation to avoid evaluation. Activate if needed. 
        trial: optuna.Trial,
        cfg: RootConfig,
        train_loader: DataLoader,
        val_loader: DataLoader, 
        emb_sizes: list[tuple[int, int]],
        device: torch.device,
        task: str = "regression"
        ) -> float:

    # Sample architecture
    #choose how many hidden layers the network has
    n_layers: int = trial.suggest_int('n_layers', *cfg.optuna.layer_range)
    
    # for each layer, choose how many neurons it has
    hidden_dims: list[int] = [
        trial.suggest_categorical(f'n_units_l{i}', cfg.optuna.units_list) for i in range(n_layers)
    ]
    dropout: float = trial.suggest_float('dropout', *cfg.optuna.dropout_range)
    # syntax: trial.suggest_float(name, low, high, log=False) # Each trial gets a different value

    # Model
    model: nn.Module = DynamicModel(
        emb_sizes, len(cfg.data.num_cols), len(cfg.data.target_cols), hidden_dims, dropout).to(device)
        
    # Optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # Loss
    criterion = (
        nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()
        )
    # Task type	                Correct loss
    # Multi-class (1 label)     CrossEntropyLoss
    # Multi-label	            BCEWithLogitsLoss
    # Multi-target regression	MSELoss

    # Loop
    # # Early stopping logic - simple code
    # patience = 5
    # best_val = float("inf")
    # epochs_no_improve = 0
    # # Early stopping logic - from a class
    early_stopping = EarlyStopping(patience=3)

    for epoch in range(cfg.optuna.n_epochs_per_trial):
        model.train()
        for xc, xn, y in train_loader:
            # Matches the model and dataloader architecture
            xc, xn, y = xc.to(device), xn.to(device), y.to(device)
            optimizer.zero_grad()
            # model(xc, xn), cat and num, not model(x)
            loss = criterion(model(xc, xn), y)
            loss.backward()
            optimizer.step()

        model.eval()
        v_loss: float = 0.0
        with torch.no_grad():
            for xc, xn, y in val_loader:
                v_loss += criterion(model(xc.to(device), xn.to(device)), y.to(device)).item()

        val_loss = v_loss / len(val_loader)
        # # Early stopping logic
        # if val_loss < best_val:
        #     best_val = val_loss
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1
        # if epochs_no_improve >= patience:
        #     break

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break

        # Optuna pruning (highly recommended) - stops bad trials automatically
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return val_loss
    