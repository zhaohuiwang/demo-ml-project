
# tests/test_early_stopping.py
import pytest
import torch.nn as nn

from demo_ml_project.utils.early_stopping import EarlyStopping


def test_early_stopping_basic():
    stopper = EarlyStopping(patience=3, min_delta=0.01, verbose=False)

    losses = [0.5, 0.4, 0.45, 0.42, 0.6, 0.7, 0.8]  # should stop after 0.8

    model = nn.Linear(1, 1)  # dummy

    for loss in losses:
        stopper(loss, model)
        if stopper.early_stop:
            break

    assert stopper.early_stop is True
    assert stopper.counter == 3
    assert stopper.best_loss == pytest.approx(0.4, abs=0.01)