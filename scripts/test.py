import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data/iris.csv')

data.head()

print(data.head())
print(data.info())
print(data.describe())

print(data['species'].value_counts())

tmp = data.drop('Id', axis=1)
g = sns.pairplot(tmp, hue='species', markers='+')
plt.show()