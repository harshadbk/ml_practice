import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules

random.seed(42)

items = ["Milk", "Bread", "Butter", "Eggs",
    "Cheese", "Juice", "Coffee", "Tea",
    "Rice", "Sugar"]

transactions = []

for i in range(20):
    transaction = random.sample(items,random.randint(2,5))
    transactions.append(transaction)

for i in transactions:
    print(i)

# encoding 

te = TransactionEncoder()

t_scaled = te.fit(transactions).transform(transactions)

print(t_scaled)

df = pd.DataFrame(t_scaled,columns=te.columns_)

# apriori

min_support = 0.20

frequent_item = apriori(df,min_support=min_support,use_colnames=True)

print(frequent_item)

frequent_item["itemsets_count"] = frequent_item["itemsets"].apply(len)

print(frequent_item)

# association rules

min_confidence = 0.60

association = association_rules(frequent_item,min_threshold=min_confidence,metric="confidence")

rules = association.sort_values(
    by=["lift","confidence"],
    ascending=False
)

print(rules)

plt.bar(frequent_item["support"],frequent_item["itemsets"].astype(str))
plt.xticks(rotation=90)
plt.title("Frequent Itemsets")
plt.xlabel("Itemsets")
plt.ylabel("Support")
plt.show()