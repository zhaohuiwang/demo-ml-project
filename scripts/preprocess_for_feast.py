
# scripts/preprocess_for_feast.py (run once)
import joblib
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

# Encoding & scaling - save fitted transformers for later use
cat_encoder = None
if cat_cols:
    cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[cat_cols] = cat_encoder.fit_transform(df[cat_cols].astype(str))

num_scaler = None
if num_cols:
    num_scaler = StandardScaler()
    df[num_cols] = num_scaler.fit_transform(df[num_cols])

tar_scaler = None
if target_cols:
    tar_scaler = StandardScaler()
    df[target_cols] = tar_scaler.fit_transform(df[target_cols])

# Entity & timestamp
if "sample_id" not in df.columns:
    df["sample_id"] = range(len(df))  # simple row index as entity ID

# Also ensure timestamp
if "event_timestamp" not in df.columns:
    df["event_timestamp"] = pd.Timestamp.now()


project_root = Path(__file__).resolve().parents[1]
output_dir = project_root / "data/processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_file_path = output_dir / "processed_for_feast.parquet"
df.to_parquet(output_file_path, index=False)
print("Saved preprocessed data to:", output_file_path)

# Save fitted preprocessing artifacts for training/inference
if cat_encoder is not None:
    joblib.dump(cat_encoder, output_dir / "cat_encoder.joblib")
    print("Saved cat_encoder.joblib")

if num_scaler is not None:
    joblib.dump(num_scaler, output_dir / "input_scaler.joblib")
    print("Saved input_scaler.joblib")

if tar_scaler is not None:
    joblib.dump(tar_scaler, output_dir / "target_scaler.joblib")
    print("Saved target_scaler.joblib")