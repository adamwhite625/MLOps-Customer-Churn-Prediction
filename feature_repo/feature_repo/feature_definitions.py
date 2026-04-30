from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

# Source pointing to the preprocessed churn dataset in Parquet format
churn_source = FileSource(
    path="../../data/processed/churn_cleaned.parquet",
    timestamp_field="event_timestamp",
)

# Primary entity representing a single customer record
customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    value_type=Int64,
)

# Feature view defining the input features for the churn prediction model
churn_feature_view = FeatureView(
    name="churn_features",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="Age", dtype=Float32),
        Field(name="Gender", dtype=Int64),
        Field(name="Tenure", dtype=Float32),
        Field(name="Usage Frequency", dtype=Float32),
        Field(name="Support Calls", dtype=Float32),
        Field(name="Payment Delay", dtype=Float32),
        Field(name="Subscription Type", dtype=Int64),
        Field(name="Contract Length", dtype=Int64),
        Field(name="Total Spend", dtype=Float32),
        Field(name="Last Interaction", dtype=Float32),
    ],
    online=True,
    source=churn_source,
)
