import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,classification_report,confusion_matrix)

from sklearn.datasets import load_iris

data = load_iris()

# print(data.data)

# print(data.target)

import seaborn as sns

df = sns.load_dataset("iris")

print(df)

X = df.iloc[:,:-1]
y = data.target

x_train,x_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state = 42
)


model = DecisionTreeClassifier(max_depth=3)

model.fit(x_train,y_train)

print(x_test)

prd = model.predict(x_test)

print(accuracy_score(y_test,prd))

print(confusion_matrix(prd,y_test))

print(classification_report(prd,y_test))


print(model.predict([[6,1,1.5,3]]))

sns.countplot(data=df,x="species")
plt.show()