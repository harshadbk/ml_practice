import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,precision_score,recall_score,confusion_matrix,classification_report)

df = pd.read_csv("heart_disease_data.csv")

print(df.head())

print(df.isnull().sum())

df["Cholesterol"].fillna(df["Cholesterol"].mean())

df.drop_duplicates()

x = df.drop(columns=["Heart_Disease"])

y = df["Heart_Disease"]

print(x)
print(y)

x_train , x_test , y_train , y_test = train_test_split(
    x,y,test_size=0.20,random_state=42
)

sc = StandardScaler()

numc = x_train.select_dtypes(include=['number']).columns

for col in numc:
    x_train[col] = sc.fit_transform(x_train[[col]]).ravel()
    x_test[col] = sc.transform(x_test[[col]]).ravel()

print(x_train.head())
print(x_test.head())

model = LogisticRegression(max_iter=1000)

model.fit(x_train,y_train)

pred = model.predict(x_test)

print("Accuracy : ",accuracy_score(y_test,pred))

prediction = model.predict([[29, 330, 280, 290, 1]])[0]

if prediction == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease Detected")

sbn.countplot(data=df , x="Exercise_Induced_Angina")
plt.show()