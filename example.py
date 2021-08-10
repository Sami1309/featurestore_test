# This is an example feature definition file

from google.protobuf.duration_pb2 import Duration

from feast import Entity, Feature, FeatureView, FileSource, ValueType

import pandas as pd

from datetime import datetime
from datetime import timezone

# Read csv data and convert to parquet

df = pd.read_csv('data/iris.csv')

timestamps = [datetime(2010,5,4,12,0,0,0,tzinfo=timezone.utc) for _ in range(len(df.index))]
created_dates = [datetime.now() for _ in range(len(df.index))]
iris_ids = [n for n in range(len(df.index))]


df['created'] = created_dates
df['event_timestamp'] = timestamps
df['iris_id'] = iris_ids

print(df.columns)
df = df.reindex(['event_timestamp', 'iris_id','sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species',
       'created'],axis=1)
print(df)

pq_dataframe = df.to_parquet('./data/iris.parquet')
#print(df)
print(pd.read_parquet('./data/iris.parquet'))
#print(pq_dataframe)

# # Read data from parquet files. Parquet is convenient for local development mode. For
# # production, you can use your favorite DWH, such as BigQuery. See Feast documentation
# # for more info.

#DOES NOT THROW ERROR IF FILE INVALID. FEAST API REQUIRES PARQUET ANYWAY
iris_measurements = FileSource(
    path=r"C:\Users\sam\Desktop\learn\feast\iris_repo\data\iris.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)

# # Define an entity for the driver. You can think of entity as a primary key used to
# # fetch features.
iris = Entity(name="iris_id", value_type=ValueType.INT64, description="iris id")

# # Our parquet files contain sample data that includes a driver_id column, timestamps and
# # three feature column. Here we define a Feature View that will allow us to serve this
# # data to our model online.

# # sepal_length,sepal_width,petal_length,petal_width,species

iris_measurements_view = FeatureView(
    name="iris_measurements",
    entities=["iris_id"],
    ttl=Duration(seconds=0),
    features=[
        Feature(name="sepal_length", dtype=ValueType.DOUBLE),
        Feature(name="sepal_width", dtype=ValueType.DOUBLE),
        Feature(name="petal_length", dtype=ValueType.DOUBLE),
        Feature(name="petal_width", dtype=ValueType.DOUBLE),
        Feature(name="species", dtype=ValueType.STRING)
    ],
    online=True,
    batch_source=iris_measurements,
    tags={},
)
