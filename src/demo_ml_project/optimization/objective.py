
# project-root(demo-ml-project)/src/demo_ml_project/optimization/objective.py

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, TimeSeriesSplit

from ..configs.schema import RootConfig
from ..data.processing import prepare_data
from ..data.dataset import InputDataset
from ..models.model import DynamicTabularModel
from ..utils.early_stopping import EarlyStopping
from ..utils.logging import get_logger

logger = get_logger(__name__)


def objective(
    trial: optuna.Trial,
    cfg: RootConfig,
    df_hpo: pd.DataFrame,                   # renamed from full_train_df
    emb_sizes: list[tuple[int, int]],
    device: torch.device,
    task: str = "regression"
) -> float:
    """
    Optuna objective function that supports optional cross-validation.
    Returns mean validation loss across folds (or from single split if CV disabled).
    """
    # ──────────────────────────────────────────────
    # 1. Sample hyperparameters
    # ──────────────────────────────────────────────
    n_layers = trial.suggest_int("n_layers", *cfg.optuna.layer_range)
    hidden_dims = [
        trial.suggest_categorical(f"n_units_l{i}", cfg.optuna.units_list)
        for i in range(n_layers)
    ]
    dropout = trial.suggest_float("dropout", *cfg.optuna.dropout_range)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    lr = trial.suggest_float("lr", *cfg.optuna.lr_range, log=True)

    # Store sampled hparams for logging / debugging if needed
    trial_hparams = {
        "n_layers": n_layers,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "optimizer": optimizer_name,
        "lr": lr,
    }

    # ──────────────────────────────────────────────
    # 2. Decide split strategy: CV or single hold-out
    # ──────────────────────────────────────────────
    if not cfg.training.cv.enabled:
        logger.debug("CV disabled → using single train/val split")
        from sklearn.model_selection import train_test_split

        train_df, val_df = train_test_split(
            df_hpo,
            test_size=cfg.training.test_size,
            random_state=cfg.training.random_state,
            shuffle=True,
        )
        folds = [(train_df, val_df)]  # single "fold"
    else:
        logger.info(f"Running {cfg.training.cv.n_folds}-fold CV ({cfg.training.cv.strategy})")

        if cfg.training.cv.strategy == "kfold":
            splitter = KFold(
                n_splits=cfg.training.cv.n_folds,
                shuffle=cfg.training.cv.shuffle,
                random_state=cfg.training.cv.random_state,
            )
        elif cfg.training.cv.strategy == "timeseries":
            splitter = TimeSeriesSplit(n_splits=cfg.training.cv.n_folds)
        else:
            raise ValueError(f"Unsupported CV strategy: {cfg.training.cv.strategy}")

        folds = [(df_hpo.iloc[train_idx], df_hpo.iloc[val_idx])
                 for train_idx, val_idx in splitter.split(df_hpo)]

    # ──────────────────────────────────────────────
    # 3. Common DataLoader settings
    # ──────────────────────────────────────────────
    loader_kwargs = dict(
        batch_size=cfg.training.batch_size,
        num_workers=0, # ← no multiprocessing during many trials
        pin_memory=(device.type != "cpu"),
        persistent_workers=False,
    )

    # ──────────────────────────────────────────────
    # 4. Evaluate across folds
    # ──────────────────────────────────────────────
    fold_val_losses = []

    for fold_idx, (train_fold_df, val_fold_df) in enumerate(folds):
        trial.set_user_attr(f"fold_{fold_idx}_status", "running")

        # Preprocess — fit on train fold only
        train_art = prepare_data(train_fold_df, cfg, fit=True)
        val_art = prepare_data(
            val_fold_df, cfg, fit=False,
            cat_encoder=train_art.cat_encoder,
            num_scaler=train_art.num_scaler,
            tar_scaler=train_art.tar_scaler,
        )

        train_loader = DataLoader(
            InputDataset(
                train_art.processed_df,
                cfg.data.cat_cols,
                cfg.data.num_cols,
                cfg.data.target_cols
            ),
            shuffle=True,
            **loader_kwargs
        )
        val_loader = DataLoader(
            InputDataset(
                val_art.processed_df,
                cfg.data.cat_cols,
                cfg.data.num_cols,
                cfg.data.target_cols
            ),
            shuffle=False,
            **loader_kwargs
        )

        # ── Create temporary model for this trial/fold
        trial_model = DynamicTabularModel(
            emb_sizes=emb_sizes,
            n_numeric=len(cfg.data.num_cols),
            n_targets=len(cfg.data.target_cols),
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)

        # ── Optimizer
        if optimizer_name == "Adam":
            optimizer = optim.Adam(trial_model.parameters(), lr=lr)
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(trial_model.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = optim.RMSprop(trial_model.parameters(), lr=lr)

        criterion = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()

        early_stopping = EarlyStopping(patience=cfg.training.patience)

        for epoch in range(cfg.optuna.n_epochs_per_trial):
            trial_model.train()
            for x_cat, x_num, y in train_loader:
                x_cat = x_cat.to(device, non_blocking=True)
                x_num = x_num.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad()
                pred = trial_model(x_cat, x_num)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

            # Validation
            trial_model.eval()
            val_loss_total = 0.0
            n_samples = 0

            with torch.no_grad():
                for x_cat, x_num, y in val_loader:
                    x_cat = x_cat.to(device, non_blocking=True)
                    x_num = x_num.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    pred = trial_model(x_cat, x_num)
                    batch_loss = criterion(pred, y).item() * len(y)
                    val_loss_total += batch_loss
                    n_samples += len(y)

            val_loss = val_loss_total / n_samples if n_samples > 0 else float("inf")

            early_stopping(val_loss, trial_model)

            if early_stopping.early_stop:
                break

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Use best loss from this fold
        fold_best_val_loss = (
            early_stopping.best_loss
            if early_stopping.best_loss is not None
            else val_loss
        )
        fold_val_losses.append(fold_best_val_loss)

        trial.set_user_attr(f"fold_{fold_idx}_best_val_rmse", fold_best_val_loss ** 0.5)

    # ──────────────────────────────────────────────
    # 5. Aggregate and report
    # ──────────────────────────────────────────────
    mean_cv_val_loss = np.mean(fold_val_losses)
    std_cv_val_loss = np.std(fold_val_losses)

    trial.set_user_attr("cv_mean_val_loss", mean_cv_val_loss)
    trial.set_user_attr("cv_mean_val_rmse", mean_cv_val_loss ** 0.5)
    trial.set_user_attr("cv_std_val_rmse", std_cv_val_loss ** 0.5)

    logger.info(
        f"Trial {trial.number} | hparams: {trial_hparams} → "
        f"CV mean val RMSE: {mean_cv_val_loss ** 0.5:.5f} (± {std_cv_val_loss ** 0.5:.5f})"
    )

    return mean_cv_val_loss