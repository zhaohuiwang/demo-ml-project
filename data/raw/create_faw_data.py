

import pandas as pd
import datetime
import numpy as np

now = datetime.datetime.now()

df = pd.DataFrame({
    "user_id": [1, 2, 3],
    "impressions_7d": [100, 50, 200],
    "clicks_7d": [10, 5, 40],
    "clicks_30d": [30, 12, 90],
    "event_timestamp": [
        now - datetime.timedelta(days=1),
        now - datetime.timedelta(days=2),
        now - datetime.timedelta(days=1),
    ],
})

df.to_parquet("data/raw/events.parquet", index=False)
print(df)
