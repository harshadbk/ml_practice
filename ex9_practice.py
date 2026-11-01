import numpy as np
import random
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules

np.random.seed(42)

items = ["Milk", "Bread", "Butter", "Eggs",
    "Cheese", "Juice", "Coffee", "Tea",
    "Rice", "Sugar"]

transactions = []

for i in range(20):
    transaction = random.sample(items,random.randint(2,5))
    transactions.append(transaction)

for i in transactions:
    print(i)

te = TransactionEncoder()

tran = te.fit(transactions).transform(transactions)

df = pd.DataFrame(tran,columns = te.columns_)

print(df)

# apriory

min_support = 0.20

frequent_itemset = apriori(df,min_support=min_support,use_colnames=True)

frequent_itemset["itemsets_size"] = frequent_itemset["itemsets"].apply(len)

print(frequent_itemset)

# association rule

min_confidence = 0.60

association = association_rules(frequent_itemset,metric="confidence",min_threshold=min_confidence)

rules = association.sort_values(by=["lift","confidence"],ascending=True)

print(rules)