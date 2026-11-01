import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,classification_report,confusion_matrix)
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
age = np.random.randint(18, 61, 1000)
salary = np.random.randint(20000, 100000, 1000)
experience = np.random.randint(1, 30, 1000)

purchase = (
    (salary > 60000) &
    (experience > 10) &
    (age > 25)
).astype(int)

df = pd.DataFrame({
    "Age": age,
    "Salary": salary,
    "Experience": experience,
    "Purchase": purchase
})

x = df[["Age","Salary","Experience"]]
y = df["Purchase"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = KNeighborsClassifier(n_neighbors=3)

model.fit(x_train, y_train)

prd = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, prd))

print(confusion_matrix(y_test, prd))
print(classification_report(y_test, prd))