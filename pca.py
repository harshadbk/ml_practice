import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = {
    "Student": [
        "S1", "S2", "S3", "S4", "S5",
        "S6", "S7", "S8", "S9", "S10",
        "S11", "S12", "S13", "S14", "S15",
        "S16", "S17", "S18", "S19", "S20"
    ],

    "Maths": [
        85, 72, 90, 65, 78,
        92, 55, 88, 70, 95,
        60, 82, 75, 98, 68,
        87, 50, 93, 77, 80
    ],

    "Science": [
        88, 75, 92, 60, 80,
        90, 58, 85, 72, 96,
        62, 84, 78, 95, 65,
        89, 52, 91, 79, 82
    ],

    "English": [
        75, 85, 80, 70, 88,
        78, 65, 90, 82, 85,
        72, 80, 87, 78, 75,
        92, 60, 83, 88, 86
    ],

    "History": [
        80, 78, 85, 68, 82,
        75, 70, 88, 76, 90,
        65, 85, 80, 92, 72,
        86, 58, 89, 83, 79
    ],

    "Computer": [
        95, 80, 98, 72, 85,
        96, 60, 92, 78, 99,
        65, 88, 82, 97, 70,
        94, 55, 95, 84, 86
    ],

    "Economics": [
        78, 82, 85, 65, 80,
        88, 62, 90, 75, 92,
        68, 84, 79, 95, 70,
        86, 58, 91, 81, 83
    ],

    "Attendance": [
        95, 85, 98, 75, 88,
        96, 70, 92, 80, 99,
        72, 90, 86, 98, 78,
        94, 65, 97, 84, 89
    ],

    "StudyHours": [
        5, 3, 6, 2, 4,
        6, 2, 5, 3, 7,
        2, 4, 3, 7, 2,
        5, 1, 6, 4, 4
    ]
}

df = pd.DataFrame(data)

print(df)

x = df.drop(columns=["Student"])

sc = StandardScaler()

scaled_x = sc.fit_transform(x)

pca = PCA(n_components=2)

x_pca = pca.fit_transform(scaled_x)

df_pca = pd.DataFrame(
    x_pca,
    columns=["PC1","PC2"]
)

print(df_pca)

print(
    "Explained Variance:",
    pca.explained_variance_ratio_
)

print(
    "Total Variance Preserved:",
    sum(pca.explained_variance_ratio_)
)

plt.scatter(x_pca[:, 0], x_pca[:, 1])

plt.show()