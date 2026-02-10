
# tests/test_dataset.py
import pytest
import torch

from demo_ml_project.data.dataset import InputDataset
from demo_ml_project.data.processing import prepare_data

def test_input_dataset(sample_df, sample_cfg):
    prep = prepare_data(sample_df, sample_cfg, fit=True)
    df_processed = prep.processed_df

    ds = InputDataset(
        df_processed,
        cat_cols=["category"],
        num_cols=["value"],
        target_cols=["target"],
    )

    assert len(ds) == len(sample_df)
    cat, num, tgt = ds[0]
    assert cat.dtype == torch.long
    assert num.dtype == torch.float32
    assert tgt.dtype == torch.float32
    assert cat.shape == (1,)
    assert num.shape == (1,)
    assert tgt.shape == (1,)