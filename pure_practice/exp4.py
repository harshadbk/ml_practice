import numpy as np
import pandas as pd

import math

import seaborn as sbn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix

# data

df = pd.read_csv(r"D:\Users\91782\Documents\WCE\V_SEM\DSA_EXPERIMENT\EXP_10\heart_disease_data.csv")

print(df.head())

x = df[["Age","Resting_Blood_Pressure","Cholesterol","Maximum_Heart_Rate","Exercise_Induced_Angina"]]
y = df["Heart_Disease"]

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.20)

le = StandardScaler()
x_train = le.fit_transform(x_train)
x_test = le.transform(x_test)

# model

model = LogisticRegression()
model.fit(x_train,y_train)

prd = model.predict(x_test)

print("Model Evaluation")

print(accuracy_score(y_test,prd))