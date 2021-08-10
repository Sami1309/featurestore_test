import pandas as pd
from datetime import datetime
from feast import FeatureStore
from feast.data_format import FileFormat



entity_df = pd.DataFrame(
    {
        "event_timestamp": [pd.Timestamp(datetime.now(), tz="UTC")],
        "iris_id": [1001]
    }
)

fs = FeatureStore(repo_path=".")

training_df = fs.get_historical_features(
    features=[
        "iris_measurements:sepal_length",
        "iris_measurements:sepal_width"
    ],
    entity_df=entity_df
).to_df()