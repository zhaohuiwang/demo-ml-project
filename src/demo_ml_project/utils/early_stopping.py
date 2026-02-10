

# src/demo_ml_project/utils/early_stopping.py

import torch


class EarlyStopping:
    """
    Early stopping handler that monitors validation loss and saves best model state.

    Stops training when validation loss stops improving after `patience` epochs.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = False):
        """
        Parameters
        ----------
        patience : int
            Number of epochs to wait after last time validation loss improved
        min_delta : float
            Minimum change in monitored quantity to qualify as improvement
        verbose : bool
            Whether to print messages when early stopping is triggered
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_loss: float | None = None
        self.best_model_state: dict | None = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        """
        Called after each validation step with current val_loss and model.

        Updates best loss/state and checks if early stopping should trigger.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            if self.verbose:
                print(f"EarlyStopping: New best val loss: {val_loss:.6f}")
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: Improved val loss to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping triggered after {self.patience} epochs without improvement")
                    
# import torch
# import numpy as np
# from pathlib import Path

# class EarlyStopping:
#     def __init__(
#             self,
#             patience: int=7,
#             verbose: bool=False,
#             delta: float=0,
#             save_model: bool=False,
#             path: str | Path ='best_model.pth'
#             ):
#         """
#         EarlyStopping if place to the end of each epoch cycle, monitors val_loss (validation loss) in each epoch (or set of epochs through DataLoader). When the criteria defined by patience and delta is met, early_stop is set to True to signal the break in the epoch iteration. save_checkpoint() is optional and maybe removed.   
#         Args:
#             patience (int): Number of epochs with no improvement before stopping.
#             verbose (bool): If True, prints messages on improvement.
#             delta (float): Minimum change to qualify as an improvement.
#             save_model (bool): Whether to save the best model.
#             path (str | Path): Path to save the model checkpoint.

#         Example:
#         early_stopping = EarlyStopping(patience=3)
#         for epoch in range(cfg.optuna.n_epochs_per_trial):
#             model.train()
#             ...
#             model.eval()
#             ...
#             val_loss = ...
#             early_stopping(val_loss, model)
#             if early_stopping.early_stop:
#                 break
        
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.inf
#         self.delta = delta
#         self.save_model = save_model
#         self.path = Path(path)

#     def __call__(self, val_loss, model):
#         score = - val_loss # lower val_loss is better

#         if self.best_score is None:
#             self.best_score = score
#             if self.save_model:
#                 self.save_checkpoint(model, val_loss)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             if self.verbose:
#                 print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             if self.save_model:
#                 self.save_checkpoint(model, val_loss)
#             self.counter = 0

#     def save_checkpoint(self, model, val_loss):
#         '''Saves model when validation loss decrease.'''
#         if self.verbose:
#             print(f"Validation loss decreased"
#                   f"({self.val_loss_min:.6f} --> {val_loss:.6f})."
#                   "Saving model ...")
#         torch.save(model.state_dict(), self.path)
#         self.val_loss_min = val_loss
