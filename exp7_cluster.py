import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# data

df = pd.read_csv("Mall_Customers.csv")
print(df.head())

df = df.drop(columns=["Gender"])

sbn.heatmap(df.corr(),annot=True)
plt.show()

print(df.head())

# preprocessing

x = df[["Annual Income (k$)","Spending Score (1-100)"]]

sc = StandardScaler()
x = sc.fit_transform(x)

print(x)

# model building

wcss = []

for i in range(2,11):
    model = KMeans(n_clusters=i,random_state=42,n_init=10)
    ans = model.fit_predict(x)
    wcss.append(model.inertia_)

sbn.lineplot(x=list(range(2,11)),y=wcss,marker="o")
plt.show()

kmc = KMeans(n_clusters=4,random_state=42,n_init=10)

result = kmc.fit_predict(x)

print(silhouette_score(x,result))

df["cluster"] = result

print(df)

sbn.scatterplot(data=df,x="Annual Income (k$)",y="Spending Score (1-100)",hue="cluster")
plt.show()
