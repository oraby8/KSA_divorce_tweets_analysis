import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string


data=pd.read_csv('enhancemntdata.csv')

sns.countplot(x='Category',data=data)
sns.countplot(x='Alies',data=data)
sns.countplot(x='Alies',hue='sentmintal',data=data)
plt.show()