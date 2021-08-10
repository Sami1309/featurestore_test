
import joblib
from sklearn.ensemble import RandomForestClassifier

from feast import FeatureStore

 
# Load the model from a file 

model_file = "./models/model.sklearn"
clf = joblib.load(model_file)

features = [
        "iris_measurements:sepal_length",
        "iris_measurements:sepal_width",
        "iris_measurements:petal_length",
        "iris_measurements:petal_width",
        "iris_measurements:species"
    ]

fs = FeatureStore(repo_path=".")

online_features = fs.get_online_features(
    features=features,
    entity_rows=[
        {"iris_id": 0},
        {"iris_id": 1}]
).to_df()

print(clf.predict(online_features[['sepal_length','sepal_width','petal_length','petal_width']]))
print(online_features[['species']])

 
# Load our features
#features = ff.online_feature_set("wine_features")
 
# Predicting quality of wine_id 0
#print(clf.predict(features["0"]))