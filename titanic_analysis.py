#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Load the Titanic dataset
df = pd.read_csv('train.csv')

# Display the first few rows of the dataset
print(df.head())


# In[4]:


# Data exploration and preprocessing
print(df.info())
print(df.describe())


# In[5]:


# Data visualization
# Survival count
survived_count = df['Survived'].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(survived_count.index, survived_count.values, color=['red', 'green'])
plt.xticks([0, 1], ['Did not survive', 'Survived'])
plt.xlabel('Survival')
plt.ylabel('Count')
plt.title('Passenger Survival Count')
plt.show()


# In[6]:


# Age distribution
plt.figure(figsize=(8, 4))
plt.hist(df['Age'].dropna(), bins=20, color='skyblue')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Passenger Age Distribution')
plt.show()


# In[7]:


# Fare distribution
plt.figure(figsize=(8, 4))
plt.hist(df['Fare'], bins=20, color='purple')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.title('Passenger Fare Distribution')
plt.show()


# In[8]:


# Correlation heatmap
plt.figure(figsize=(8, 6))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[9]:


# Data analysis and insights
# Survival rate by sex
survival_rate_sex = df.groupby('Sex')['Survived'].mean()
print(survival_rate_sex)


# In[10]:


# Survival rate by cabin class
survival_rate_class = df.groupby('Pclass')['Survived'].mean()
print(survival_rate_class)


# In[11]:


# Survival rate by age group
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 18, 30, 50, 80], labels=['Child', 'Young Adult', 'Adult', 'Elderly'])
survival_rate_agegroup = df.groupby('AgeGroup')['Survived'].mean()
print(survival_rate_agegroup)


# In[ ]:




