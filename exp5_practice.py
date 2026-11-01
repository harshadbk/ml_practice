import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score,precision_score,f1_score,
classification_report,confusion_matrix,recall_score)

# import data

df = pd.read_csv("credit_risk_knn_dataset.csv")

print(df.head())

print(df.isna().sum())

# x and y 

x = df[["Debt_to_Income_Ratio","Credit_Score","Annual_Income","Total_Assets"]]
y = df["Credit_Risk_Rating"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train,y_train)

prd = model.predict(x_test)

# classification Report

print(accuracy_score(y_test,prd))

sbn.heatmap(df.corr(),annot=True)
plt.show()