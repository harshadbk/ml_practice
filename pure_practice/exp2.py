import numpy as np
import pandas as pd

import math

import seaborn as sbn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df = pd.read_csv(r"D:\Users\91782\Documents\WCE\V_SEM\DSA_EXPERIMENT\EXP_10\Mall_Customers.csv")

print(df.head())

print(df.info())
print(df.isna().sum())

df.drop_duplicates()

print(df)

sbn.scatterplot(data=df,x="Annual Income (k$)",y="Spending Score (1-100)")
plt.show()

x = df[["Annual Income (k$)","Age"]]
y = df["Spending Score (1-100)"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42
)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model = LinearRegression()
model.fit(x_train, y_train)

prd = model.predict(x_test)

print("MAE :", mean_absolute_error(y_test, prd))
print("RMSE:", math.sqrt(mean_squared_error(y_test, prd)))
print("R2 :", r2_score(y_test, prd))