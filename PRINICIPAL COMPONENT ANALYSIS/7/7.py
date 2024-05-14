#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df = pd.read_csv('cancer_tumor_data_features.csv')


# In[6]:


df.head()


# In[ ]:





# In[7]:


from sklearn.preprocessing import StandardScaler


# In[8]:


scaler = StandardScaler()


# In[9]:


scaled_X = scaler.fit_transform(df)


# In[10]:


scaled_X


# In[ ]:





# In[11]:


# Because we scaled the data, this won't produce any change.
# We've left if here because you would need to do this for unscaled data
scaled_X -= scaled_X.mean(axis=0)


# In[12]:


scaled_X


# In[ ]:





# In[19]:


# Grab Covariance Matrix
covariance_matrix = np.cov(scaled_X, rowvar=False)


# In[20]:


# Get Eigen Vectors and Eigen Values
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)


# In[21]:


# Choose som number of components
num_components=2


# In[22]:


# Get index sorting key based on Eigen Values
sorted_key = np.argsort(eigen_values)[::-1][:num_components]


# In[23]:


# Get num_components of Eigen Values and Eigen Vectors
eigen_values, eigen_vectors = eigen_values[sorted_key], eigen_vectors[:, sorted_key]


# In[24]:


# Dot product of original data and eigen_vectors are the principal component values
# This is the "projection" step of the original points on to the Principal Component
principal_components=np.dot(scaled_X,eigen_vectors)


# In[25]:


principal_components


# In[ ]:





# In[26]:


plt.figure(figsize=(8,6))
plt.scatter(principal_components[:,0],principal_components[:,1])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[ ]:





# In[31]:


from sklearn.datasets import load_breast_cancer


# In[32]:


# REQUIRES INTERNET CONNECTION AND FIREWALL ACCESS
cancer_dictionary = load_breast_cancer()


# In[33]:


cancer_dictionary.keys()


# In[34]:


cancer_dictionary['target']


# In[ ]:





# In[35]:


plt.figure(figsize=(8,6))
plt.scatter(principal_components[:,0],principal_components[:,1],c=cancer_dictionary['target'])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[ ]:




