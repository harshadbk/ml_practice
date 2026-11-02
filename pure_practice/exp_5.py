import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score
)

# Load Dataset
df = pd.read_csv(r"D:\Users\91782\Documents\WCE\V_SEM\DSA_EXPERIMENT\EXP_10\heart_disease_data.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Features and Target
X = df[[
    "Age",
    "Resting_Blood_Pressure",
    "Cholesterol",
    "Maximum_Heart_Rate"
]]

y = df["Heart_Disease"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.55,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN Model
model = KNeighborsClassifier(
    n_neighbors=5,
    metric="minkowski",
    p=2
)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy :", accuracy_score(y_test, y_pred))

print("Precision:",precision_score(y_test,y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))