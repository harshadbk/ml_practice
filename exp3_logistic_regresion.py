import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score,precision_score,f1_score,
classification_report,confusion_matrix,recall_score)

df = pd.read_csv("heart_disease_data.csv")

print(df.head())

print(df.isnull())

print(df.isna().sum())

x = df[["Age","Resting_Blood_Pressure","Cholesterol","Maximum_Heart_Rate","Exercise_Induced_Angina"]]
y = df["Heart_Disease"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

sc = StandardScaler()

cols_to_scale = x_train.columns[:-1]

x_train[cols_to_scale] = sc.fit_transform(x_train[cols_to_scale])
x_test[cols_to_scale] = sc.transform(x_test[cols_to_scale])

model = LogisticRegression()

model.fit(x_train,y_train)

prd = model.predict(x_test)

print("Model Evaluation :")

print(accuracy_score(y_test,prd)," \n ")

print(precision_score(y_test,prd)," \n ")

print(recall_score(y_test,prd)," \n ")

print(classification_report(y_test,prd)," \n ")

print(confusion_matrix(y_test,prd)," \n ")

sbn.heatmap(df.corr(),annot=True)
plt.show()