# featurestore_test

Implemented following tutorials on https://docs.feast.dev/ using the iris dataset.

Generates a feature store, trains a model and serves local inference requests.

## TODO

kubernetes implementation for online store

helper functions for data translation requests

automatic label paramaterization

### Insights

#### Advantages of Feast:

concise declarative framework for registering feature store

#### Disadvantages of Feast

No automatic label parameter for features

requires timestamp without obvious automatic generation for datasets not served via a static timescale

Unclear about feature reqtrieval, serving consistency

Online materalization imprecise and should be by default within scripting framework.