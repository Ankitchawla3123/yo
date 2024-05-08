import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("datafile.csv")

def validation(row):
    violations=[]
    if not (0 <= row['Age'] <= 150):
        violations.append('Age rule not followed')
    if row['Age']<=row['yearsmarried']:
        violations.append('2nd point violated')
    if row["status"] not in ["married", "single", "widowed"]:
        violations.append("Status should be married, single, or widowed.")
    if (row['Age']<18 and row['agegroup']!='child') or (18<= row['Age']<= 65 and row['agegroup']!='adult') or (row['Age'] >65 and row['agegroup']!='elderly'):
        violations.append('Last violation')
    return violations

df['Violations']=df.apply(validation,axis=1)
countviloations=df['Violations'].apply(len).sum()
print("Total number of violations: ",countviloations)
countviloationseachrow=df['Violations'].apply(len)

plt.bar(range(len(countviloationseachrow)),countviloationseachrow)

plt.xlabel("Row number")
plt.ylabel("Number of violations")
plt.show()