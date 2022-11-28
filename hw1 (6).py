#!/usr/bin/env python
# coding: utf-8

# In[19]:


# HW1
# Tom Adoni
# STA 4724

# Problem 1

def largerIndex(c):
    # Initialize empty list to append to
    k= []
    # Initialize a length variable for original list
    l = len(c)
    # Loop through original list to append to new list
    for i in range(0,l):
        # Check which number needs to be appended
        if (c[i] == i):
            k.append(0)
        elif (c[i] > i):
            k.append(1)
        else:
            k.append(-1)  
    # Return the new list
    return k
l1 = [1,2,0,4,2,1,40,-5]
largerIndex(l1)


# In[20]:


l2 = [0,3,2,1,32,3,4,0]
largerIndex(l2)


# In[3]:


# Problem 2

def squareUpTo(n):
    # Initialize int variable to keep track of numbers being squared
    i = 0
    # Initialize empty list to store the squares
    l = []
    # Loop that runs until the square is bigger than the input
    while i**2 <= n:
        l.append(i**2)
        i=i+1
    # Return the list of squares
    return l

squareUpTo(10)


# In[4]:


squareUpTo(100)


# In[18]:


# Problem 3
import random

def flip1in3():
    # Assign a random integer between 0 and 1 twice, add the sum, if it's 0, try again, for 1, return false, for 2, return true
    x = 0
    # 2/3 chance for false, 1/3 chance for true
    for i in range(2):
        x+=random.randint(0,1)
    if(x == 0):
        for i in range(2):
            x+=random.randint(0,1)
    if(x == 1):
        return False
    else:
        return True
flip1in3()


# In[ ]:


# Problem 4 

def duplicates(c):
    # Initialize the length variable
    l = len(c)
    # Initialize an empty list for duplicates
    x = []
    # Nested for loop to compare every number in the list and check for duplicates
    for i in range(l):
        k = i + 1
        for j in range(k, l):
            if c[i] == c[j] and c[i] not in x:
                x.append(c[i])
    # Return the list of duplicates
    return x

l3 = [1,2,5,3,6,2,4,5]
duplicates(l3)


# In[ ]:


l4 = [1,3,5,5,1,4,3]
duplicates(l4)


# In[ ]:




