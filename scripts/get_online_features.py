from pprint import pprint
from feast import FeatureStore

store = FeatureStore(repo_path=".")

feature_vector = store.get_online_features(
    features=[
        "iris_measurements:sepal_width"
    ],
    entity_rows=[{"iris_id": 0}],
).to_dict()

pprint(feature_vector)