#!/usr/bin/env python
# coding: utf-8

# ### STA 4724: Homework 3 - Due Friday, Oct. 21 
# **Instructions**: Finsih the assign by directly answering the question or finishing the code in this Jupyter notebook. 
# 
# After you finish, submit the saved notebook to webcourses.

# **Your name:**   Tom Adoni                  
# 
# **Your ID:**    4860378

# ## Question 1: kNN
# For the *yeast* dataset, we want to predicte the compartment in a cell that a yeast protein will localize to based on properties of its sequence.
# 
# **(a)** Read the training and testing datasets.

# In[2]:


import pandas as pd
ye_train = pd.read_csv('yeast_train.csv')
ye_test = pd.read_csv('yeast_test.csv')


# **(b)** Use leave-one-out cross validation to select the value of $k$ for kNN model.
# 
# Hint 1: try to apply kNN with a fixed $k$ first, then figure out how to use cross validation. In *yeast_3.txt*, I list the result when $k=3$ and you can use it as a reference.
# 
# Hint 2: Some of the $k$ values are not suitable to be the "best". You shouldn't even test them in the cross validation.

# In[5]:


# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# Create feature and target arrays
X = ye_train.data
y = ye_train.target
  
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
  
knn = KNeighborsClassifier(n_neighbors=7)
  
knn.fit(X_train, y_train)
  
# Predict on dataset which model has not seen before
print(knn.predict(X_test))


# **(c)** Train the kNN model with the best $k$, and use it to predicte the testing data.

# In[ ]:


neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
  
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
      
    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
  
# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


# **(d)** Print the confusion matrix, then manually compute the accuracy based on the confusion matrix. That is, don't call any built-in function for the accuracy, but type the equation you will use for calculating the accuracy.

# In[ ]:


fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# ## Question 2: Logistic Regression
# 
# We study the *myopia* dataset in this problem.
# 
# **(a)** Read the data file and make a scatterplot of MYOPIA vs. SPHEQ.

# In[ ]:


import pandas as pd
mp = pd.read_csv('myopia.csv')


# **(b)** Fit the logistic regression model of SPHEQ on MYOPIA

# In[ ]:


import numpy
from sklearn import linear_model

#Reshaped for Logistic function.
X = mp.loc[:,"SPHEQ"]
y = mp.loc[:,"MYOPIC"]

logr = linear_model.LogisticRegression()
logr.fit(X,y)


# **(c)** Plot the logistic function found in (b) together with the scatterplot. You are looking for something like ![image.png](attachment:image.png)
# 
# Hint: Although we didn't make this logistic function plot in class, the idea behind it is similar to the linear function plot in the earlier lecture. 

# In[ ]:


#predict if tumor is cancerous where the size is 3.46mm:
predicted = logr.predict(numpy.array([3.46]).reshape(-1,1))
print(predicted)


# ## Question 3: Naive Bayes
# We have the *vote* datasets collected from 1984 United States Congressional Voting Records.
# 
# **(a)** Read the training and testing data file, and make any necessary preprocess so the data is ready to be used by naive bayes.
# 
# 
# Hint: computer cannot read 'y/n' just like it cannot read tweets directly.

# In[ ]:


import pandas as pd
vo_train = pd.read_csv('vote_train.csv')
vo_test = pd.read_csv('vote_test.csv')


# **(b)** Train the bayes model and use it to predict the voting result on the testing data.

# In[ ]:


from sklearn.naive_bayes import GaussianNB
import numpy as np

target_names = np.array(['Positives','Negatives'])

# add columns to your data frame
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75
df['Type'] = pd.Factor(targets, target_names)
df['Targets'] = targets

# define training and test sets
train = df[df['is_train']==True]
test = df[df['is_train']==False]

trainTargets = np.array(train['Targets']).astype(int)
testTargets = np.array(test['Targets']).astype(int)

# columns you want to model
features = df.columns[0:7]

# call Gaussian Naive Bayesian class with default parameters
gnb = GaussianNB()

# train model
y_gnb = gnb.fit(train[features], trainTargets).predict(train[features])


# **(c)** Plot the ROC curve and report AUC.

# In[ ]:


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# method II: ggplot
from ggplot import *
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')


# **(d, Extra question, No bonus point)** Based on the data in the traning set, what's the most significant character for the republican voters? Can you write a short program to find it out? 
# 
# Hint: We didn't cover this in class.

# **Answer:** 
