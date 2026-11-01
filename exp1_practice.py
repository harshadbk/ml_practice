import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
import random
random.seed(42)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules
from sklearn.decomposition import PCA

data = {
    "CustomerID":[101,102,103,104,105,106,107],
    "Age":[23,45,np.nan,31,22,54,38],
    "Annual_Income_k":[50,85,62,np.nan,45,120,350],
    "City":["New York","Paris","New York","London",np.nan,"London","Paris"]
}

df = pd.DataFrame(data)

print(df.head())

print(df.info())

print(df.isna().sum())

# fill numerical values with mean

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Annual_Income_k"] = df["Annual_Income_k"].fillna(df["Annual_Income_k"].median())

# fill categorial values with mode

df["City"] = df["City"].fillna(df["City"].mode()[0])

print(df)

# outliers detection using iqr

mini = df["Annual_Income_k"].quantile(0.25)
maxi = df["Annual_Income_k"].quantile(0.75)

IQR = maxi - mini

minir = mini - 1.5 * IQR
maxir = maxi + 1.5 * IQR

df["Annual_Income_k"] = np.where(df["Annual_Income_k"] > maxir , maxir,df["Annual_Income_k"])

print(df)

# Label encoding

le = LabelEncoder()
mmc = MinMaxScaler()

df["City"] = le.fit_transform(df["City"])
df[["Age","Annual_Income_k"]] = mmc.fit_transform(df[["Age","Annual_Income_k"]])

print(df)

# plots and graphs

sbn.heatmap(df.corr(), annot=True)
plt.show()