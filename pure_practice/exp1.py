import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

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

# imputation

# mean imputation

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Annual_Income_k"] = df["Annual_Income_k"].fillna(df["Annual_Income_k"].median())

print(df.isna().sum())

df["City"] = df["City"].fillna(df["City"].mode()[1])

print(df.isna().sum())

# outlier detection

q1 = df["Annual_Income_k"].quantile(0.25)
q3 = df["Annual_Income_k"].quantile(0.75)

IQR = q3-q1

upper = q3 + IQR * 1.5
lower = q1 - IQR * 1.5

df["Annual_Income_k"] = np.where(df["Annual_Income_k"] > upper , upper,df["Annual_Income_k"])

print(df["Annual_Income_k"])

# Encoding

le = LabelEncoder()
sc = StandardScaler()

df["City"] = le.fit_transform(df["City"])
df[["Annual_Income_k","Age"]] = sc.fit_transform(df[["Annual_Income_k","Age"]])

print(df)

df = df.drop(columns=["City","CustomerID"])

sbn.heatmap(df.corr(),annot=True)
plt.show()

sbn.scatterplot(data = df,x="Annual_Income_k",y="Age")
plt.show()