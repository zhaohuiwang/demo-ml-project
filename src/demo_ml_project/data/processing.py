
#project-root(demo-ml-project)/src/demo_ml_project/data/processing.py

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from ..configs.schema import RootConfig
from ..configs.artifacts import PreprocessingArtifacts

def prepare_data(df: pd.DataFrame, cfg: RootConfig) -> PreprocessingArtifacts:
    """ 
    A data preparation function:
    - Encodes categorical features
    - Scales numeric features and targets

    Parameters
    ----------
    df: pd.DataFrame
    cfg : a Pydantic Model

    Returns
    -------
    Tuple[pd.DataFrame, OrdinalEncoder, StandardScaler, StandardScaler]
        Processed dataframe, fitted categorical encoder, fitted feature scaler,
        fitted target scaler
    """

    # cfg.data.cat_cols, cfg.data.num_cols, cfg.data.target_cols
    # <class 'omegaconf.listconfig.ListConfig'>
    # Pandas' __setitem__ (the code behind df[some_key] = value) checks isinstance(key, list)
    # So need to convert them to list
    data_cat_cols: list[str] = list(cfg.data.cat_cols)
    data_num_cols: list[str] = list(cfg.data.num_cols)
    data_target_cols: list[str] = list(cfg.data.target_cols)

    cat_encoder: OrdinalEncoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
        )
    df[data_cat_cols] = cat_encoder.fit_transform(df[data_cat_cols].astype(str))

    num_scaler: StandardScaler = StandardScaler()
    df[data_num_cols] = num_scaler.fit_transform(df[data_num_cols])

    tar_scaler: StandardScaler = StandardScaler()
    df[data_target_cols] = tar_scaler.fit_transform(df[data_target_cols])

    return PreprocessingArtifacts(
            processed_df=df,
            cat_encoder=cat_encoder,
            num_scaler=num_scaler,
            tar_scaler=tar_scaler,
        )