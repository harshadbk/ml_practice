import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# data

df = pd.read_csv(r"D:\Users\91782\Documents\WCE\V_SEM\DSA_EXPERIMENT\EXP_10\Mall_Customers.csv")
print(df.head())

x = df[["Annual Income (k$)","Spending Score (1-100)"]]

sc = StandardScaler()
x_scaled = sc.fit_transform(x)

wcss = []

for i in range(2,11):
    model = KMeans(n_clusters=i,n_init=10,random_state=42)
    model.fit_predict(x_scaled)
    wcss.append(model.inertia_)

sbn.lineplot(x=list(range(2,11)),y=wcss,marker='o')
plt.title("within cluster sum of sqaure")
plt.xlabel("No of cluster")
plt.ylabel("Score")
plt.show()

model = KMeans(n_clusters=7,n_init=10,random_state=42)

cluster = model.fit_predict(x_scaled)

df["cluster"] = cluster

print(df.head())

print("silhouette Score",silhouette_score(x_scaled,cluster))

sbn.scatterplot(  
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="cluster"
)

plt.show()