import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Automatic Feature Creation using pandas
df = pd.DataFrame(data={'Gender': [['Female','Male'][np.random.randint(0,2)] for i in range(0,50000)],
                        'Height': [np.random.randint(140, 190) for i in range(0,50000)],
                        'Weight': [np.random.randint(40,120) for i in range(0,50000)],
                        'Index': [np.random.randint(1,6) for i in range(0,50000)]
                        })

#df = pd.read_csv('C:/Users/user/Downloads/archive/500_Person_Gender_Height_Weight_Index.csv')

df.head()
df.info()
df['Index']=df['Index'].astype('object')

for col in df.select_dtypes(include='object'):
    print('\nColumn',col)
    print('Unique entries',df[col].nunique())
    print(df[col].value_counts())

def change_func(x):
    if x >= 180:
        return '5'
    elif x < 180 and x >= 170:
        return '4'
    elif x < 170 and x >= 160:
        return '3'
    elif x < 160 and x >= 150:
        return '2'
    else:
        return '1'

df['Index'] = df['Height'].apply(lambda x : change_func(x))
prep_df = df.describe()
prep_df

prep_mean = prep_df['Height']['mean']
prep_std = prep_df['Height']['std']
# one hot encoding
Target_Index = pd.get_dummies(df['Index'], columns = ['Index'], prefix= 'Index')
Target_Index.head()

# standardization data
df['mean_adjusted'] = df['Height'].apply(lambda x: x - prep_mean)
df['standardized'] = df['mean_adjusted'].apply(lambda x: x / prep_std)

# UNIVARIATE ANALYSIS
import matplotlib.pyplot as plt
import seaborn as sns

for col in df.iloc[:,1:3]:
    fig, ax= plt.subplots(1,3, figsize=(25,5))
    sns.boxplot(df[col], ax=ax[0])
    sns.violinplot(df[col], ax=ax[1])
    sns.distplot(df[col], ax=ax[2], label='Skewness %.2f%%'%(df[col].skew()))
    plt.legend(loc='upper left')
    plt.show()

# BIVARIATE ANALYSIS
fig, ax= plt.subplots(1,2, figsize=(15,8))
sns.barplot(df['Gender'],df['Height'], ax=ax[0])
sns.lineplot(df['Gender'],df['Height'], ax=ax[1])
plt.show()

# The mean height of females is slightly more than that of males
fig, ax= plt.subplots(1,2, figsize=(15,8))
sns.barplot(df['Gender'],df['Weight'], ax=ax[0])
sns.lineplot(df['Gender'],df['Weight'], ax=ax[1])
plt.show()
# The mean weight of Males is slightly more than that of females

# ANALYSIS WITH TARGET
fig, ax= plt.subplots(1,2, figsize=(15,8))
sns.lineplot(df['Index'],df['Weight'], ax=ax[0])
sns.lineplot(df['Index'],df['Height'], ax=ax[1])
plt.show()

tab = pd.crosstab(index=df['Index'], columns=df['Gender'])
tab.plot(kind="bar", stacked=False, figsize=(15,5))
plt.xlabel('Index')
plt.ylabel('Count')
plt.legend()
plt.show()

# Females are more Extremely Weak than males
# Males are more Weak than females
# Females are more Normal than males
# Females are more Overweight than males
# Females are more Obese than males
# Males are more Extreme Obese than females

plt.figure(figsize=(15,8))
sns.scatterplot(x='Height', y='Weight', hue='Index', data=df)
plt.show()


sns.pairplot(df, diag_kind='kde', hue='Index')
plt.show()