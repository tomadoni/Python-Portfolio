#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Tom Adoni

import pandas as pd

df = pd.read_csv('WI_BC_data.csv')


# In[6]:


df.head()


# In[7]:


import numpy as np
import matplotlib.pyplot as plt


# In[11]:


df['diagnosis'].value_counts().plot(kind='bar')


# In[13]:


# 2/3 of the tumors are benign, while the rest are malignant


# In[ ]:




