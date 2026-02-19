
# scripts/preprocess_for_feast.py (run once)

import pandas as pd

from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from demo_ml_project.configs.training.schema import DataConfig

conf = DataConfig.load_default()

num_cols = conf.num_cols
cat_cols = conf.cat_cols
target_cols = conf.target_cols

df = pd.read_parquet("data/processed/model_df.parquet")

# Cleaning
df = df.drop(columns=conf.drop_columns, errors="ignore")

if "population" in df.columns:
    df["population"] = df["population"].fillna(df["population"].median()).astype("int32")

df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Encoding & scaling
if cat_cols:
    df[cat_cols] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(df[cat_cols].astype(str))

if num_cols:
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])

if target_cols:
    df[target_cols] = StandardScaler().fit_transform(df[target_cols])

# Entity & timestamp
if "sample_id" not in df.columns:
    df["sample_id"] = range(len(df))  # simple row index as entity ID

# Also ensure timestamp
if "event_timestamp" not in df.columns:
    df["event_timestamp"] = pd.Timestamp.now()


project_root = Path(__file__).resolve().parents[1]
output_file_path = project_root / "data/processed/processed_for_feast.parquet" 

df.to_parquet(output_file_path, index=False)
print("Saved to:", output_file_path)