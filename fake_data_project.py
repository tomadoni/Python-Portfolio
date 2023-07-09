#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import pandas as pd


# In[7]:


# Set product names
products = ["Garrett Wilson Men's Jersey", "Sauce Gardner Men's Jersey", "New York Jets Men's Cap",
            "Breece Hall Men's Jersey", "Aaron Rodgers Men's Jersey", "Quinnen Williams Men's Jersey",
           "C.J. Mosley Men's Jersey"]

# Set customer names
customers = ["Joe Johnson", "Deron Williams", "Michael Gandolfini", "Woody Johnson", "Miakl Bridges",
             "Joe Tsai", "Dwayne Johnson"]

# Create the dataframe

df = pd.DataFrame(columns = ["customer_name", "product", "revenue"])


# In[8]:


# Generate 1,000 rows of fake data

for i in range(1000):
    # Select a random product
    product = random.choice(products)
    
    # Select a random customer
    customer = random.choice(customers)
    
    # Generate a random revenue amount
    revenue = random.randint(100, 10000)
    
    # Add the data to the DataFrame
    df = df.append({"customer_name": customer, "product" : product, "revenue" : revenue}, ignore_index = True)


# In[9]:


df.head()


# In[ ]:




