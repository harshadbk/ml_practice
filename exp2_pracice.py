import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_absolute_error,r2_score,mean_squared_error
import math

np.random.seed(42)

n_students = 50

study_hours = np.random.uniform(1.0, 10.0, n_students)

marks = 30 + (study_hours * 6.5) + np.random.normal(0, 5, n_students)
marks = np.clip(marks, 0, 100)

data = {
    "Study_Hours": study_hours,
    "Marks": marks
}

df = pd.DataFrame(data)

x = df[["Study_Hours"]]
y = df["Marks"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)

# sc = StandardScaler()

# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

print(x_train)
print(x_test)

model = LinearRegression()

model.fit(x_train,y_train)

prd = model.predict(x_test)

# model evaluation

print(mean_absolute_error(y_test,prd))
print(mean_squared_error(prd,y_test))

sns.heatmap(df.corr(),annot=True)
plt.show()