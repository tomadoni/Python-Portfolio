#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Load the dataset
df = pd.read_csv('Housing.csv')


# In[ ]:


# Summary statistics of the numerical columns
df.describe()

# Count plots of the categorical columns
for col in df.select_dtypes(include = 'object').columns:
    sns.countplot(x = col, data = df)
    plt.show


# In[ ]:


# Correlation matrix of the numerical columns
sns.heatmap(df.select_dtypes(exclude = 'object').corr(), annot = True)


# In[ ]:


# Box plots of the numerical columns by target
for col in df.select_dtypes(exclude = 'object').columns:
    sns.boxplot(x = 'price', y = col, data = df)
    plt.show()


# In[ ]:


# Bar plots of the categorical columns by target
for col in df.select_dtypes(exclude = 'object').columns:
    sns.barplot(x = 'price', y = col, data = df)
    plt.show()


# In[ ]:




