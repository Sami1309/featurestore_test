from datetime import datetime

import pandas as pd

from feast import FeatureStore

from sklearn.model_selection import train_test_split

store = FeatureStore(repo_path=".")

#TODOway to get store size from reference
print(store.list_entities())

entity_df = pd.DataFrame(
    {
        "event_timestamp": [pd.Timestamp(datetime.now(), tz="UTC") for _ in range(150)],
        "iris_id": [n for n in range(150)]
    }
)



training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_measurements:sepal_length",
        "iris_measurements:sepal_width",
        "iris_measurements:petal_length",
        "iris_measurements:petal_width",
        "iris_measurements:species"
    ],
).to_df()



print(training_df)

X=training_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]  # Features
y=training_df['species']  # Labels

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(clf.predict([[3, 5, 4, 2]])
)

import pandas as pd
feature_imp = pd.Series(clf.feature_importances_,index=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']).sort_values(ascending=False)
print(feature_imp)

import joblib


model_file = "./models/model.sklearn"
joblib.dump(clf, model_file)
