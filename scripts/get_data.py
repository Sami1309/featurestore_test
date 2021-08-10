from datetime import datetime

import pandas as pd

from feast import FeatureStore

entity_df = pd.DataFrame.from_dict(
    {
        "iris_id": [0],
        "event_timestamp": [pd.Timestamp(datetime.now(), tz="UTC")],
    }
)

store = FeatureStore(repo_path=".")

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_measurements:sepal_length",
        "iris_measurements:sepal_width",
    ],
).to_df()

print(training_df.head())