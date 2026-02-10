
# tests/test_data_processing.py
import pytest
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from demo_ml_project.data.processing import prepare_data


def test_prepare_data_fit(sample_df, sample_cfg):
    result = prepare_data(sample_df, sample_cfg, fit=True)

    df = result.processed_df
    assert set(df.columns) == {"category", "value", "target", "extra"}
    assert df["category"].dtype.kind in "i"  # integer
    assert df["value"].mean().round(6) == pytest.approx(0.0, abs=1e-6)
    assert abs(df["value"].std() - 1.0) < 0.2   # relaxed for small sample

    assert isinstance(result.cat_encoder, OrdinalEncoder)
    assert isinstance(result.num_scaler, StandardScaler)
    assert isinstance(result.tar_scaler, StandardScaler)

    # Check scaling actually happened
    assert df["value"].mean().round(6) == pytest.approx(0.0, abs=1e-6)
    assert df["value"].std().round(3) == pytest.approx(1.0, abs=0.15)  # ← relaxed


def test_prepare_data_transform(sample_df, sample_cfg):
    train_df = sample_df.iloc[:3]
    fit_art = prepare_data(train_df, sample_cfg, fit=True)

    result = prepare_data(
        sample_df,
        sample_cfg,
        fit=False,
        cat_encoder=fit_art.cat_encoder,
        num_scaler=fit_art.num_scaler,
        tar_scaler=fit_art.tar_scaler,
    )
    df = result.processed_df

    assert df["category"].max() >= 0
    assert abs(df["value"].mean()) < 1.5   # transformed using train stats only

def test_prepare_data_missing_fitted_objects(sample_df, sample_cfg):
    with pytest.raises(ValueError, match="all fitted encoders/scalers must be provided"):
        prepare_data(sample_df, sample_cfg, fit=False)