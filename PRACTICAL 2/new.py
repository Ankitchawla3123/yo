import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#PART 1
df=pd.read_csv('iris_dirty.csv')
dfwithoutna=df.dropna()
print("Percentage of complete observation",len(dfwithoutna)/len(df)*100)

# PART 2
specialval=['?','NA','NaN','-']
df.replace(specialval,np.nan,inplace=True)
# rules=file.readlines()
# rulescorrect=[]
# for i in rules:
#     rulescorrect.append(i.strip())

with open('rules.txt', 'r') as file:
    rules_content = file.readlines()

rules_content1=[]
for i in rules_content:
    rules_content1.append(i.strip())


print("Rules defined in the text file:")
print(rules_content)

    
def checkrules(row,rule):
    if rule == "Species should be one of the following values setosa, versicolor or virginica.":
        return not(pd.notna(row['Species']) and row['Species'].lower() in ['setosa', 'versicolor', 'virginica'])
    elif rule == "All measured numerical properties of an iris should be positive.":
        return not(pd.notna(row['Species']) and all(row.drop('Species') > 0))
    elif rule == "The petal length of an iris is at least 2 times its petal width.":
        return not(pd.notna(row['Species']) and row['Petal.Length'] >= 2 * row['Petal.Width'])
    elif rule == "The sepal length of an iris cannot exceed 30 cm.":
        return not(pd.notna(row['Species']) and row['Sepal.Length'] <= 30)
    elif rule == "The sepals of an iris are longer than its petals.":
        return not(pd.notna(row['Species']) and row['Sepal.Length'] > row['Petal.Length'])

# rules_content={} 
# print(rulescorrect)
# rule_break_count = {}

rule_break_count = {}
for rule in rules_content1:
    if rule:
        rule_break_count[rule] = sum(df.apply(checkrules, rule=rule, axis=1))

# print(rule_break_count.keys())
for i in rule_break_count:
    print(i,':- ', rule_break_count[i])
    

plt.bar(rule_break_count.keys(), rule_break_count.values())
plt.xticks(rotation=45)
plt.show()

# print(rule_break_count)


Q1 = df['Sepal.Length'].quantile(0.25)

# Calculate the third quartile (Q3)
Q3 = df['Sepal.Length'].quantile(0.75)

# Calculate the interquartile range (IQR)
IQR = Q3 - Q1

# Define the lower bound for outliers
lower_bound = Q1 - 1.5 * IQR

# Define the upper bound for outliers
upper_bound = Q3 + 1.5 * IQR

# Filter the DataFrame to include only outliers
sepal_length_outliers = df[(df['Sepal.Length'] < lower_bound) | (df['Sepal.Length'] > upper_bound)]

# Print the outliers
print("Outliers in Sepal Length:")
print(sepal_length_outliers)



