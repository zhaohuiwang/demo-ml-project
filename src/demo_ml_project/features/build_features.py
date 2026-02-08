
import pandas as pd


def compute_user_stats(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()

    df["ctr_7d"] = df["clicks_7d"] / df["impressions_7d"].clip(lower=1)
    df["click_count_30d"] = df["clicks_30d"]
    df["created_at"] = pd.Timestamp.utcnow()

    return df[
        [
            "user_id",
            "ctr_7d",
            "click_count_30d",
            "event_timestamp",
            "created_at",
        ]
    ]

