
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, FeatureService
from feast.types import Float32, Int64
from feast.value_type import ValueType
from feast.data_format import ParquetFormat

sample_entity = Entity(
    name="sample_id",
    value_type=ValueType.INT64,
    description="Unique sample identifier"
)

raw_source = FileSource(
    path="../../data/processed/processed_for_feast.parquet",
    event_timestamp_column="event_timestamp",
    file_format=ParquetFormat(),
)

raw_view = FeatureView(
    name="raw_features",
    entities=[sample_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name=col, dtype=Int64) for col in ["sex_code", "age_group", "states_abbr", "sample_id"]
    ] + [
        Field(name=col, dtype=Float32) for col in [
            "year", "population", "real_gdp", "real_personal_income", "real_pce",
            "gdp", "personal_income", "disposable_personal_income", "pce",
            "regional_price_parities", "n_jobs", "regional_price_deflator", "non_smoker_rate",
            "breast", "lung_and_bronchus", "melanoma_of_the_skin"
        ]
    ],
    source=raw_source,
)

training_service = FeatureService(name="training_features", features=[raw_view])
inference_service = FeatureService(name="inference_features", features=[raw_view])