import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")

print(df.head())

x = df[["Annual Income (k$)","Spending Score (1-100)"]]

sc = StandardScaler()

x_scaled = sc.fit_transform(x)

print(x_scaled)

wcss = []

for i in range(2,11):
    model = KMeans(n_clusters=i,random_state=42,n_init=10)
    clust = model.fit_predict(x_scaled)
    wcss.append(model.inertia_)

sns.lineplot(x=list(range(2,11)),y=wcss,marker="o")
plt.show()

model = KMeans(n_clusters=5,random_state=42,n_init=10)
result = model.fit_predict(x_scaled)

print("silhouette_score",silhouette_score(x_scaled,result))

df["cluster"] = result

print(df.head())

sns.scatterplot(data = df,x="Annual Income (k$)",y="Spending Score (1-100)",hue="cluster")
plt.show()