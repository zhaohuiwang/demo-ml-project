
# scripts/preprocess_for_feast.py (run once)

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

df = pd.read_parquet("data/processed/model_df.parquet")

# Cleaning
df = df.drop(columns=["states", "fips", "sex"], errors="ignore")
if "population" in df.columns:
    df["population"] = df["population"].fillna(df["population"].median()).astype("int32")

num_cols = df.select_dtypes("number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = ["sex_code", "age_group", "states_abbr"]
num_cols = [
    "year", "population", "real_gdp", "real_personal_income", "real_pce",
    "gdp", "personal_income", "disposable_personal_income", "pce",
    "regional_price_parities", "n_jobs", "regional_price_deflator", "non_smoker_rate"
]
target_cols = ["breast", "lung_and_bronchus", "melanoma_of_the_skin"]

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

df.to_parquet("data/processed/processed_for_feast.parquet", index=False)
print("Preprocessed data saved to data/processed/processed_for_feast.parquet")


# from pathlib import Path
# project_root = Path(__file__).resolve().parents[2]  # up 2 levels from scripts/
# df.to_parquet(project_root / "data/processed/processed_for_feast.parquet", index=False)
# print("Saved to:", project_root / "data/processed/processed_for_feast.parquet")