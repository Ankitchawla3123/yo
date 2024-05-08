import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]

# Q1.a) Calculate mean of attributes for each species
mean_df = iris_df.groupby('species').mean()
# Q1.b) Calculate sum of attributes for each species
sum_df = iris_df.groupby('species').sum()

# Q1.c) Calculate standard deviation of attributes for each species
std_dev_df = iris_df.groupby('species').std()

# Q2) Randomly sample 80% of the records to create a new dataset Train_iris
train_iris, _ = train_test_split(iris_df, test_size=0.2, random_state=42)

# Q3) Discretize Petal.length and Petal.width into three categories: low, medium, high

train_iris['Petal.length_category'] =pd.cut(train_iris['petal length (cm)'], bins=3, labels=['low', 'medium', 'high'])
train_iris['Petal.width_category'] = pd.cut(train_iris['petal width (cm)'], bins=3, labels=['low', 'medium', 'high'])

# Displaying the results
print("Mean of attributes for each species:")
print(mean_df)
print("\nSum of attributes for each species:")
print(sum_df)
print("\nStandard deviation of attributes for each species:")
print(std_dev_df)
print("\nTrain_iris dataset with discretized Petal.length and Petal.width:")
print(train_iris.head())
