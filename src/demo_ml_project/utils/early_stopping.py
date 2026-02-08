
import torch
import numpy as np
from pathlib import Path

class EarlyStopping:
    def __init__(
            self,
            patience: int=7,
            verbose: bool=False,
            delta: float=0,
            save_model: bool=False,
            path: str | Path ='best_model.pth'
            ):
        """
        EarlyStopping if place to the end of each epoch cycle, monitors val_loss (validation loss) in each epoch (or set of epochs through DataLoader). When the criteria defined by patience and delta is met, early_stop is set to True to signal the break in the epoch iteration. save_checkpoint() is optional and maybe removed.   
        Args:
            patience (int): Number of epochs with no improvement before stopping.
            verbose (bool): If True, prints messages on improvement.
            delta (float): Minimum change to qualify as an improvement.
            save_model (bool): Whether to save the best model.
            path (str | Path): Path to save the model checkpoint.

        Example:
        early_stopping = EarlyStopping(patience=3)
        for epoch in range(cfg.optuna.n_epochs_per_trial):
            model.train()
            ...
            model.eval()
            ...
            val_loss = ...
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break
        
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_model = save_model
        self.path = Path(path)

    def __call__(self, val_loss, model):
        score = - val_loss # lower val_loss is better

        if self.best_score is None:
            self.best_score = score
            if self.save_model:
                self.save_checkpoint(model, val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_model:
                self.save_checkpoint(model, val_loss)
            self.counter = 0

    def save_checkpoint(self, model, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f"Validation loss decreased"
                  f"({self.val_loss_min:.6f} --> {val_loss:.6f})."
                  "Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
