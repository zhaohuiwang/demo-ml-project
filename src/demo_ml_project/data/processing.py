
#project-root(demo-ml-project)/src/demo_ml_project/data/processing.py

# src/demo_ml_project/data/processing.py

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from ..configs.training.schema import RootConfig
from ..configs.training.artifacts import PreprocessingArtifacts


def prepare_data(
    df: pd.DataFrame,
    cfg: RootConfig,
    fit: bool = True,
    cat_encoder: OrdinalEncoder | None = None,
    num_scaler: StandardScaler | None = None,
    tar_scaler: StandardScaler | None = None,
) -> PreprocessingArtifacts:
    """
    Preprocess tabular data with support for separate fit and transform phases
    to prevent data leakage during cross-validation or hold-out evaluation.

    - When `fit=True`: fits encoders/scalers on the input data
    - When `fit=False`: applies previously fitted transformers

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing categorical, numerical and target columns
    cfg : RootConfig
        Configuration object containing column names
    fit : bool, default=True
        Whether to fit new transformers (True) or only transform (False)
    cat_encoder, num_scaler, tar_scaler : fitted objects or None
        Pre-fitted transformers required when fit=False

    Returns
    -------
    PreprocessingArtifacts
        Container with processed dataframe and fitted transformers

    Raises
    ------
    ValueError
        If fit=False but any required fitted transformer is missing
    """
    # Work on a copy to avoid modifying the input
    df_processed = df.copy()

    # Extract column lists (convert from OmegaConf/ListConfig if needed)
    cat_cols = list(cfg.data.cat_cols)
    num_cols = list(cfg.data.num_cols)
    target_cols = list(cfg.data.target_cols)

    if fit:
        # Fit new transformers
        cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=int,
        )
        df_processed[cat_cols] = cat_encoder.fit_transform(
            df_processed[cat_cols].astype(str)
        )

        num_scaler = StandardScaler()
        df_processed[num_cols] = num_scaler.fit_transform(df_processed[num_cols])

        tar_scaler = StandardScaler()
        df_processed[target_cols] = tar_scaler.fit_transform(df_processed[target_cols])

    else:
        # Transform only — require pre-fitted objects
        if any(obj is None for obj in (cat_encoder, num_scaler, tar_scaler)):
            raise ValueError(
                "When fit=False, all fitted encoders/scalers must be provided"
            )

        df_processed[cat_cols] = cat_encoder.transform(
            df_processed[cat_cols].astype(str)
        )
        df_processed[num_cols] = num_scaler.transform(df_processed[num_cols])
        df_processed[target_cols] = tar_scaler.transform(df_processed[target_cols])

    return PreprocessingArtifacts(
        processed_df=df_processed,
        cat_encoder=cat_encoder,
        num_scaler=num_scaler,
        tar_scaler=tar_scaler,
    )

# import pandas as pd
# from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# from ..configs.schema import RootConfig
# from ..configs.artifacts import PreprocessingArtifacts

# def prepare_data(
#         df: pd.DataFrame,
#         cfg: RootConfig,
#         fit: bool = True,
#         cat_encoder: OrdinalEncoder = None,
#         num_scaler: StandardScaler = None,
#         tar_scaler: StandardScaler = None
#         ) -> PreprocessingArtifacts:
#     """ 
#     A data preparation function:
#     - Encodes categorical features
#     - Scales numeric features and targets

#     Parameters
#     ----------
#     df: pd.DataFrame
#     cfg: a Pydantic Model
#     fit: True if fit mode otherwise transform mode
#     ...

#     Returns
#     -------
#     Tuple[pd.DataFrame, OrdinalEncoder, StandardScaler, StandardScaler]
#         Processed dataframe, fitted categorical encoder, fitted feature scaler,
#         fitted target scaler
#     """
#     df = df.copy()

#     # cfg.data.cat_cols, cfg.data.num_cols, cfg.data.target_cols
#     # <class 'omegaconf.listconfig.ListConfig'>
#     # Pandas' __setitem__ (the code behind df[some_key] = value) checks isinstance(key, list)
#     # So need to convert them to list
#     data_cat_cols: list[str] = list(cfg.data.cat_cols)
#     data_num_cols: list[str] = list(cfg.data.num_cols)
#     data_target_cols: list[str] = list(cfg.data.target_cols)

#     if fit:
#         cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
#         df[data_cat_cols] = cat_encoder.fit_transform(df[data_cat_cols].astype(str))
        
#         num_scaler = StandardScaler()  # Or RobustScaler() for outliers
#         df[data_num_cols] = num_scaler.fit_transform(df[data_num_cols])
        
#         tar_scaler = StandardScaler()
#         df[data_target_cols] = tar_scaler.fit_transform(df[data_target_cols])
#     else:
#         # Transform only
#         df[data_cat_cols] = cat_encoder.transform(df[data_cat_cols].astype(str))
#         df[data_num_cols] = num_scaler.transform(df[data_num_cols])
#         df[data_target_cols] = tar_scaler.transform(df[data_target_cols])
    
#     return PreprocessingArtifacts(
#         processed_df=df,
#         cat_encoder=cat_encoder,
#         num_scaler=num_scaler,
#         tar_scaler=tar_scaler,
#     )