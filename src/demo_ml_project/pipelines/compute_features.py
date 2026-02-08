import pandas as pd
from my_ml_project.features.build_features import compute_user_stats


def run():
    events = pd.read_parquet("data/raw/events.parquet")
    features = compute_user_stats(events)

    features.to_parquet(
        "data/processed/user_stats.parquet",
        index=False,
    )

    print(features)


if __name__ == "__main__":
    run()

# python -m my_ml_project.pipelines.compute_features
# Now you have: data/processed/user_stats.parquet
# Tell Feast about this table (update path): feast/data_sources/warehouse.py