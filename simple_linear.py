import numpy as np
import pandas as pd
from sklearn.preprocessing import (StandardScaler,LabelEncoder)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (mean_absolute_error,r2_score,mean_squared_error,accuracy_score,precision_score,recall_score,classification_report,confusion_matrix)

df = pd.read_csv("StudentsPerformance.csv")

print(df.head(5))

print(df.isnull().sum())

df.drop_duplicates(inplace=True)

x = df.drop(columns=["math score"])

y = df["math score"]

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.20,random_state=42
)

lb = LabelEncoder()

categorical_cols = x_train.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    x_train[col] = lb.fit_transform(x_train[col])
    x_test[col] = lb.transform(x_test[col])


print(x_train.head())
print(y_train.head())

print(x_test.head())
print(y_test.head())

model = LinearRegression()

# model = SGDRegressor(max_iter = 10,random_state=42)

model.fit(x_train,y_train)

prd = model.predict(x_test)

act = np.round(prd,2)
pr = np.round(y_test,2)

print("Evaluation Matrix")

print(pd.DataFrame({
    "Actual":act,
    "Predicted":pr
}).head(10))

print("Mean Absolute Error ",mean_absolute_error(y_test,prd))
print("R Square Value ",r2_score(y_test,prd))