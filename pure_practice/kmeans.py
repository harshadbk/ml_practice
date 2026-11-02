import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import random

# data 

df = pd.read_csv(r"D:\Users\91782\Documents\WCE\V_SEM\DSA_EXPERIMENT\EXP_10\Mall_Customers.csv")

print(df.head())

x = df[["Annual Income (k$)","Spending Score (1-100)"]]

print(x)

# scaling

sc = StandardScaler()

x_scaled = sc.fit_transform(x)

print(x_scaled)

wcss = []

for i in range(2,11):
    model = KMeans(n_clusters=i,random_state=42,n_init=10)
    model.fit_predict(x_scaled)
    wcss.append(model.inertia_)

sbn.lineplot(x=list(range(2,11)),y=wcss,marker='o')
plt.show()

# actual model

model = KMeans(n_clusters=7,random_state=42,n_init=10)
cluster = model.fit_predict(x_scaled)

df["cluster"] = cluster

print(silhouette_score(x_scaled,cluster))

print(df.head())

sbn.scatterplot(data=df,x="Annual Income (k$)",y="Spending Score (1-100)",hue="cluster")
plt.show()