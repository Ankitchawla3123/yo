from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

dataset1 = pd.DataFrame({
    'TransactionID': [1, 2, 3, 4, 5],
    'Items': [['Milk', 'Bread', 'Butter'], ['Milk', 'Butter'], ['Bread', 'Butter'], ['Milk', 'Bread'], ['Milk', 'Butter']]
})

dataset2 = pd.DataFrame({
    'TransactionID': [1, 2, 3, 4, 5],
    'Items': [['Coffee', 'Tea'], ['Coffee', 'Juice'], ['Tea', 'Juice'], ['Coffee', 'Tea'], ['Coffee', 'Tea', 'Juice']]
})

def preprocess_data(dataset):
    # Apply one-hot encoding directly using Pandas' get_dummies() function
    one_hot_encoded = dataset['Items'].str.join('|').str.get_dummies()
    
    return one_hot_encoded

def evaluate_apriori(dataset, min_support, min_confidence):
    dataset_processed = preprocess_data(dataset)
    frequent_itemsets = apriori(dataset_processed, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    correctness = frequent_itemsets['support'] * rules['confidence']
    return correctness

# Parameters for dataset 1
min_support_a = 0.5
min_confidence_a = 0.75

# Parameters for dataset 2
min_support_b = 0.6
min_confidence_b = 0.6

# Evaluate dataset 1
correctness_a = evaluate_apriori(dataset1, min_support_a, min_confidence_a)

# Evaluate dataset 2
correctness_b = evaluate_apriori(dataset2, min_support_b, min_confidence_b)

print("\nCorrectness of patterns for dataset 1:")
print(correctness_a)

print("\nCorrectness of patterns for dataset 2:")
print(correctness_b)
