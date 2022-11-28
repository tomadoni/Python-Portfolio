#!/usr/bin/env python
# coding: utf-8

# ## STA 4724: Homework 2 - Due Wednesday, Sep. 28 
# **Instructions**: Finsih the assign by directly answering the question or finishing the code in this Jupyter notebook. 
# 
# After you finish, submit the saved notebook to webcourses.
# 
# 

# ## Question 1
# The cost of the maintenance of a certain type of tractor seems to increase with age. The file
# *tractor.csv* contains ages (years) and 6-monthly maintenance costs for n = 17 such tractors.
# 
# **(a)** Read the data file.

# In[4]:


# Tom Adoni
import pandas as pd
df = pd.read_csv("tractor.csv")


# **(b)** Create a scatterplot of tractor maintenance cost versus age.

# In[6]:


import matplotlib.pyplot as plt
df.plot(kind='scatter',x='age',y='cost')


# **(c)** Using regression to fit the model:
# $$\text{cost}=\beta_0+\beta_1 \text{age}$$
# in two different ways.
# 
# Firstly, use python's bulit-in linear regression solver.

# In[19]:


import numpy as np
from sklearn.linear_model import LinearRegression
x = df['age']
x = x.values.reshape((-1, 1))
y = df['cost']
model = LinearRegression()
model.fit(x, y)
r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")
x_new = np.arange(5).reshape((-1, 1))
x_new
y_new = model.predict(x_new)
y_new


# Secondly, use close form solution of least square.

# In[18]:


from scipy import optimize
plt.style.use('seaborn-poster')
# generate x and y
x = df['age']
y = df['cost']
# assemble matrix A
A = np.vstack([x, np.ones(len(x))]).T

# turn y into a column vector
y = y[:, np.newaxis]
# Direct least square regression
alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)
print(alpha)

# plot the results
plt.figure(figsize = (10,8))
plt.plot(x, y, 'b.')
plt.plot(x, alpha[0]*x + alpha[1], 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# **(d)** Add both fitted lines (in different color/style) to the scatterplot. They should give you the same solution.

# In[31]:


df.plot(kind='scatter',x='age',y='cost')
plt.plot(x, alpha[0]*x + alpha[1], 'r', color = 'purple')


# **(e)** Suppose you are considering buying a tractor  that is three years old, what would you expect your 6-monthly maintenance costs to be?

# ###### **Answer**:  4500

# ## Question 2.1
# Let's generate a synthetic dataset for regression! It should have m = 150 data points and each has n = 75 dimensions (features).
# 
# **(a)** Set the random seed to be 0.

# In[ ]:


import random

random.seed(0)


# **(b)** Let $X\in\mathbb{R}^{m\times n}$ be a random matrix using *numpy.random.rand()* function.

# In[32]:


import numpy as np
np.random.rand(75,150)


# **(c)** Set the first 10 components of $\hat{\beta}$ to be some random values between $-10$ and $10$, and all the other components to zero. 
# 
# Hint: Read what *numpy.random.rand()* generates, and how you generate random numbers in a different interval.

# In[34]:


sampl = np.random.uniform(low=-10, high=10, size=(0,10))


# **(d)** Computer $Y=X\hat{\beta}+\varepsilon$ where $\varepsilon\in\mathbb{R}^{m\times 1}$ is a random noise vector generated using *numpy.random.randn()* with mean 0 and standard deviation 0.1.

# In[35]:


0 + 0.1 * np.random.randn(75, 150)


# ## Question 2.2
# We know the problem we generated above has a sparse solution. So we should solve it with lasso regression.
# 
# **(a)** Solve the lasso regression problem with $\lambda=0.00001$.

# In[60]:


from sklearn.linear_model import Lasso

x_all_seeds = np.append(np.ones(len(x)),x[:])

for num in range(150):
    for num in range(75):
        x_all_seeds = np.append(random.randint(0,1))
    
x_all_seeds = np.reshape(x_all_seeds,(75,150)).T

lasso_reg = Lasso(alpha=0.00001, normalize=True) 
lasso_reg.fit(x_all_seeds,y)
y_pred = lasso_reg.predict(x_all_seeds)

plt.scatter(x, y)
plt.plot(x, y_pred)
plt.show()


# **(b)** Use 10-folder cross validation to find the best regularization parameter between $0$ and $1$.

# In[71]:


# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# create dataset
X, y = make_classification(n_samples=150, n_features=75, n_redundant=0, random_state=1)
# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model = LogisticRegression()
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# **(c)** Solve the lasso regression problem again with the best regularization parameter.

# In[76]:


import numpy as np # Numpy is a Matlab-like package for array manipulation and linear algebra
import pandas as pd # Pandas is a data-analysis and table-manipulation tool
import urllib.request # Urlib will be used to download the dataset

# Import the function that performs sample splits from scikit-learn
from sklearn.model_selection import train_test_split

# Load the output variable with pandas (download with urllib if not downloaded previously)

localAddress = 'tractor.csv'
try:
    y = pd.read_csv(localAddress, header=None)
except:
    y = pd.read_csv(localAddress, header=None)
y = y.values # Transform y into a numpy array

# Print some information about the output variable
print('Class and dimension of output variable:')
print(type(y))
print(y.shape)

# Load the input variables with pandas 
localAddress = 'tractor.csv'
try:
    x = pd.read_csv(localAddress, header=None)
except:
    x = pd.read_csv(localAddress, header=None)
x = x.values

# Print some information about the input variables
print('Class and dimension of input variables:')
print(type(x))
print(x.shape)

# Create the training sample
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.4, random_state=1)

# Split the remaining observations into validation and test
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=1) 

# Print the numerosities of the three samples
print('Numerosities of training, validation and test samples:')
print(x_train.shape[0], x_val.shape[0], x_test.shape[0])


# **(d)** Compare the estimated $\beta_{\text{est}}$ with the ground truth $\hat{\beta}$ by computing $\|\beta_{\text{est}}-\hat{\beta}\|_2$. The distance should be small.

# In[79]:


def generate_data(noise_frac):
  X = np.random.rand(ntrials,nneurons)
  X = np.random.normal(size=(ntrials,nneurons))
  
  beta = np.random.randn(nneurons)
  y = X @ beta

  # not very important how I generated noise here
  noise_x = np.random.multivariate_normal(mean=zeros(nneurons), cov=diag(np.random.rand(nneurons)), size=ntrials)

                            
  X_noise = X + noise_x*noise_frac

  return X_noise, y, beta
  


# ## Question 3
# In 1988, US cattle producers voted on whether or not to each pay a dollar per head towards the marketing campaigns of the American Beef Council. To understand the vote results, the Montana state cattlemen's association looked at the effect of the physical size of the farm and the value of the farms' gross revenue on voter preference. *beef.csv* consist of the vote results (YES in %), average SIZE of farm (in hundred acres), and average VAL of products sold annually by each farm (in thousand dollors) for each of Montana’s 56 counties.
# 
# **(a)** Read the data file.

# In[80]:


beefds = read_csv('beef.csv')


# **(b)** Use Multivariate Linear Regression to fit YES with SIZE and log(VAL) as the regressors.

# In[83]:


X = beefds[['SIZE', 'VAL']]
y = beefds['YES']
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X, y)
predictedVote = regr.predict([[54,853]])

print(predictedVote)


# **(c)** Is this a good fit? Numerically check via a sutiable statistics test.

# In[90]:


import scipy.stats as stats
stats.ttest_ind(beefds['SIZE'][beefds['VAL'] > 500],
                beefds['SIZE'][beefds['VAL'] < 500])


# **(d)** In this dataset, what fact may potentially be a probelm for our regression analysis?
# 
# Hint: Does the effect of SIZE change depending on log(VAL)?

# **Answer:** Yes

# In[ ]:




