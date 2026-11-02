import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier,plot_tree,export_text
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix

# data imports 

df = pd.read_csv("credit_risk_knn_dataset.csv")

print(df.head())

x = df[["Debt_to_Income_Ratio","Credit_Score","Annual_Income","Total_Assets"]]
y = df["Credit_Risk_Rating"]

# train test split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

# feature scaling

sc = StandardScaler()

x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

print(x_test_scaled)
print(x_train_scaled)

# model selection

dt = DecisionTreeClassifier(max_depth=5,random_state=42)

dt.fit(x_train,y_train)

# prediction

prd = dt.predict(x_test)

# model evaluation

print("Model Evaluation")

print("Accuracy Score ",accuracy_score(y_test,prd))

print("precision Score",precision_score(y_test,prd,average='micro'))

print("Recall Score",recall_score(y_test,prd,average='micro'))

print("Confusion Matrix",confusion_matrix(y_test,prd))

# sample prediction

df = pd.DataFrame(
    [[0.7,900,100000,23000]],
    columns=["Debt_to_Income_Ratio","Credit_Score","Annual_Income","Total_Assets"]
)

tree_rules = export_text(dt,feature_names=list(x.columns))
print(tree_rules)

x = df[["Debt_to_Income_Ratio","Credit_Score","Annual_Income","Total_Assets"]]
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x)

print(dt.predict(x_train_scaled))

