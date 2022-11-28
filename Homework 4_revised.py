#!/usr/bin/env python
# coding: utf-8

# ### STA 4724: Homework 4 - Due Monday, Nov. 14 
# **Instructions**: Finsih the assign by directly answering the question or finishing the code in this Jupyter notebook. 
# 
# After you finish, submit the saved notebook to webcourses.

# **Your name:**       Tom Adoni              
# 
# **Your ID:**         4860378

# ## Question 1: Hierarchical Clustering
# Use USArrests dataset and perform hierarchical clustering on the states.
# 
# This data set contains statistics, in arrests per 100,000 residents for assault, murder, and rape in each of the 50 US states in 1973. Also given is the percent of the population living in urban areas.
# 
# **(a)** Read the dataset.

# In[1]:


import pandas as pd
arrests = pd.read_csv('USArrests.csv')


# **(b)** Using hierarchical clustering with Euclidean distance to cluster the states.

# In[ ]:


from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist

X = arrests.data

X_dist = pdist(X)
X_link = linkage(X, method='ward')


# **(c)** Plot the dendrogram that shows the last 10 merged clusters.

# In[ ]:


import matplotlib.pyplot as plt
dendrogram(X_link, truncate_mode = 'lastp',p=10, show_contracted = True)
plt.show()


# **(d)** Cut the dendrogram at a height that results in three distinct clusters. Which states belong to each of the three clusters?
# 
# Code a program to show your answer.

# In[ ]:


from scipy.cluster.hierarchy import fcluster
max_d = 9
clusters = fcluster(X_link, max_d, criterion='distance')

print(clusters)


# ## Question 2: Decision Trees
# 
# Boston is a data set containing housing values in 506 suburbs of Boston. We would like to predict the house price using this dataset.
# 
# Here is the data descriptions: 
# 
# * crim: 
# per capita crime rate by town.
# 
# * zn: 
# proportion of residential land zoned for lots over 25,000 sq.ft.
# 
# * indus: 
# proportion of non-retail business acres per town.
# 
# * chas: 
# Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# 
# * nox: 
# nitrogen oxides concentration (parts per 10 million).
# 
# * rm: 
# average number of rooms per dwelling.
# 
# * age: 
# proportion of owner-occupied units built prior to 1940.
# 
# * dis: 
# weighted mean of distances to five Boston employment centres.
# 
# * rad: 
# index of accessibility to radial highways.
# 
# * tax: 
# full-value property-tax rate per $10,000.
# 
# * ptratio: 
# pupil-teacher ratio by town.
# 
# * lstat: 
# lower status of the population (percent).
# 
# * medv: 
# median value of owner-occupied homes in $1000s.
# 
# Hint: When you build a decision tree to predict house prices, each left node ends up with several houses in it. The average price of these houses is the predicted price for this left node.  You can view this as a supervised clustering, and the prediction is the mean of each cluster. This is also known as Decision Tree Regressor.
# 
# For this question, you can either build the regular decision tree and then take the mean for each node by yourself. Or you can use the DecisionTreeRegressor function that we didn't cover in class.
# 
# **(a)** Load the dataset.

# In[ ]:


import pandas as pd
boston = pd.read_csv('Boston.csv')


# **(b)** Use cross-validation to find the best parameters (max_depth, min_samples_leaf) for the decision tree.

# In[ ]:


X = boston[['rm','indus','chas']]
Y = boston['medv']

import numpy as np
import sklearn.model_selection as ms
from sklearn import tree
from sklearn.model_selection import GridSearchCV

XTrain, XTest, YTrain, YTest = ms.train_test_split(X, Y, test_size= 0.3, random_state=1)

depth_val = np.arange(2,11)
leaf_val = np.arange(1,31, step=9)

grid_s = [{'max_depth': depth_val,'min_samples_leaf': leaf_val}]
model = tree.DecisionTreeClassifier(criterion='entropy')

cv_tree = GridSearchCV(estimator=model,param_grid=grid_s,cv=ms.KFold(n_splits=10))
cv_tree.fit(XTrain, YTrain)

best_depth = cv_tree.best_params_['max_depth']

best_min_samples = cv_tree.best_params_['min_samples_leaf']

print(best_depth, best_min_samples)


# **(c)** Train the decision tree with the best parameters, then generate a Graphviz visualization of the tree.
# 
# Hint 1: You will generate a graphviz .dot file first. Install Graphviz or use online tool to make the actual plot, then you can save or screenshot the tree plot.
# 
# Hint 2: To insert a image in the Jupyter notebook, click *Edit -> Insert Image*.

# In[ ]:


model = tree.DecisionTreeClassifier(criterion='entropy',max_depth=best_depth,min_samples_leaf=best_min_samples)

BostonTree = model.fit(XTrain, YTrain)

survive_pred = BostonTree.predict(XTest)

survive_proba = BostonTree.predict_proba(XTest)

tree.export_graphviz(BostonTree, out_file='BostonTree.dot', max_depth=3, feature_names=X.columns, class_names=['House','Price'])


# Post the tree plot here: 
