import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
import random

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules

random.seed(42)

items = ["Milk", "Bread", "Butter", "Eggs",
    "Cheese", "Juice", "Coffee", "Tea",
    "Rice", "Sugar"]

transactions = []

for i in range(0,len(items)+1):
    transaction = random.sample(items,random.randint(2,5))
    transactions.append(transaction)

for i in transactions:
    print(i)

# encoding

te = TransactionEncoder()
itemsets = te.fit(transactions).transform(transactions)
print(itemsets)

# apriori

df = pd.DataFrame(itemsets,columns = te.columns_)

# min support

min_support = 0.20

apr = apriori(df,min_support=min_support,use_colnames=True)

print(apr)

apr["itemsets_count"] = apr["itemsets"].apply(len)

print(apr)

# association rule 

min_confidence = 0.60

asso = association_rules(apr,min_threshold=min_confidence,metric="confidence")

rules = asso.sort_values(
    by=["lift","confidence"],
    ascending=False
)

print(rules)