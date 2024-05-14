#!/usr/bin/env python
# coding: utf-8

# #Principal Component Analysis (PCA) - 
# Linear scikit-learn Doc
# 
# scikit-learn Parameters
# 
# Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
# 1901 by Karl Pearson
# 
# Unsupervised Machine Learning
# Wikipedia
# 
# Statistical procedure that utilise orthogonal transformation technology
# 
# Convert possible correlated features (predictors) into linearly uncorrelated features (predictors) called principal components
# 
# # of principal components <= number of features (predictors)
# 
# First principal component explains the largest possible variance
# 
# Each subsequent component has the highest variance subject to the restriction that it must be orthogonal to the preceding components.
# 
# A collection of the components are called vectors.
# 
# Sensitive to scaling
# 
# Note:
# 
# Used in exploratory data analysis (EDA)
# 
# Visualize genetic distance and relatedness between populations.
# 
# Method:
# 
# Eigenvalue decomposition of a data covariance (or correlation) matrix
# 
# Singular value decomposition of a data matrix (After mean centering / normalizing ) the data matrix for each attribute.
# 
# Output
# 
# Component scores, sometimes called factor scores (the transformed variable values)
# 
# loadings (the weight)
# 
# Data compression and information preservation
# 
# Visualization
# 
# Noise filtering
# 
# Feature extraction and engineering

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
sns.set()


# In[ ]:





# In[2]:


rnd_num = np.random.RandomState(42)
X = np.dot(rnd_num.rand(2,2), rnd_num.randn(2, 500)).T


# In[3]:


X


# In[ ]:





# In[4]:


X[:, 0] = - X[:, 0]


# In[ ]:





# In[5]:


plt.scatter(X[:, 0], X[:, 1]);
plt.axis('equal');


# In[ ]:





# In[6]:


from sklearn.decomposition import PCA


# In[7]:


pca = PCA(n_components=2)
pca.fit(X)


# In[8]:


print(pca.components_)


# In[9]:


print(pca.explained_variance_)


# In[10]:


print(pca.explained_variance_ratio_)


# In[11]:


plt.scatter(X[:, 0], X[:, 1], alpha=0.3)


# plot data

for k, v in zip(pca.explained_variance_, pca.components_):
    vec = v * 3 * np.sqrt(k)
    
    ax = plt.gca()
    arrowprops=dict(arrowstyle='<-',
                    linewidth=4,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', pca.mean_, pca.mean_ + vec, arrowprops=arrowprops)
    ax.text(-0.90, 1.2,'PC1', ha='center', va='center', rotation=-42, size=12)
    ax.text(-0.1,-0.6,'PC2', ha='center', va='center', rotation=50, size=12)
plt.axis('equal');


# # Dimensionality Reduction with PCA

# In[12]:


pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)


# In[13]:


X.shape


# In[14]:


X_pca.shape


# In[15]:


X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2);
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal');


# In[ ]:




