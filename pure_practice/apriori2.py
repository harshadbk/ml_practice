import numpy as np
import pandas as pd
import random

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules

random.seed(42)

items = [
    "Milk", "Bread", "Butter", "Eggs",
    "Cheese", "Juice", "Coffee", "Tea",
    "Rice", "Sugar"
]

transactions = []

for i in range(20):
    transaction = random.sample(items,random.randint(2,5))
    transactions.append(transaction)

for i in transactions:
    print(i)

# encoding 

te = TransactionEncoder()

te_scale = te.fit(transactions).transform(transactions)

print(te_scale)

df = pd.DataFrame(te_scale,columns = te.columns_)

print(df)

# apriori

min_support = 0.20

frequent_item = apriori(df,min_support=min_support,use_colnames=True)

frequent_item["itemsets_count"] = frequent_item["itemsets"].apply(len)

print(frequent_item)

# association rules

min_confidence = 0.60

rule = association_rules(frequent_item,metric="confidence",min_threshold=min_confidence)

rules = rule.sort_values(
    by=["lift","confidence"],
    ascending = False
)

print(rules)