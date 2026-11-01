import random
import pandas as pd

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

for i, t in enumerate(transactions, start=1):
    print(f"T{i} :", t)

te = TransactionEncoder()

te_array = te.fit(transactions).transform(transactions)

df = pd.DataFrame(te_array,columns=te.columns_)

print(df)

# frequent itemsets

min_support = 0.20

frequent_itemset = apriori(df,min_support=min_support,use_colnames=True)

frequent_itemset["itemsets_size"] = frequent_itemset["itemsets"].apply(len)

print(frequent_itemset)

# confidence level 

min_confidence = 0.60

rules = association_rules(
    frequent_itemset,
    metric="confidence",
    min_threshold=min_confidence
)

rules = rules.sort_values(
    by=["lift","confidence"],
    ascending=False
)

print("\nRules :",rules)