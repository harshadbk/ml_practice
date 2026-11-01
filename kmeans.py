import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load Dataset
df = pd.read_csv("heart_disease_data.csv")

# Display Dataset
print(df.head())
print(df.info())

x = df[["Age","Resting_Blood_Pressure","Cholesterol","Maximum_Heart_Rate","Exercise_Induced_Angina"]]

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)

wcss = []

for i in range(2,11):
    model = KMeans(n_clusters=i,random_state=42,n_init=10)
    model.fit(x_scaled)
    wcss.append(model.inertia_)

sbn.lineplot(x=list(range(2, 11)), y=wcss, marker="o")
plt.show()

kmeans = KMeans(n_clusters=6,random_state=42,n_init=5)

cluster = kmeans.fit_predict(x_scaled)

df["cluster"] = cluster

score = silhouette_score(x_scaled, cluster)
print("Silhouette Score :", score)

print(df.groupby("cluster").mean())

# Final Dataset
print(df.head())

sbn.scatterplot(
    data=df,
    x="Age",
    y="Cholesterol",
    hue="cluster"
)

plt.title("Heart Disease Clusters")
plt.show()