#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('cancer_tumor_data_features.csv')


# In[3]:


df.head()


# In[5]:


from sklearn.preprocessing import StandardScaler


# In[6]:


scaler = StandardScaler()


# In[7]:


scaled_X = scaler.fit_transform(df)


# In[9]:


scaled_X


# In[ ]:





# In[10]:


from sklearn.decomposition import PCA


# In[11]:


help(PCA)


# In[ ]:





# In[12]:


pca = PCA(n_components=2)


# In[14]:


principal_components = pca.fit_transform(scaled_X)


# In[15]:


plt.figure(figsize=(8,6))
plt.scatter(principal_components[:,0],principal_components[:,1])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[ ]:





# In[16]:


from sklearn.datasets import load_breast_cancer


# In[17]:


# REQUIRES INTERNET CONNECTION AND FIREWALL ACCESS
cancer_dictionary = load_breast_cancer()


# In[18]:


cancer_dictionary.keys()


# In[19]:


cancer_dictionary['target']


# In[ ]:





# In[20]:


plt.figure(figsize=(8,6))
plt.scatter(principal_components[:,0],principal_components[:,1],c=cancer_dictionary['target'])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[ ]:





# In[21]:


pca.n_components


# In[22]:


pca.components_


# In[ ]:





# In[23]:


df_comp = pd.DataFrame(pca.components_,index=['PC1','PC2'],columns=df.columns)


# In[ ]:





# In[24]:


df_comp 


# In[ ]:





# In[25]:


plt.figure(figsize=(20,3),dpi=150)
sns.heatmap(df_comp,annot=True)


# In[ ]:





# In[26]:


pca.explained_variance_ratio_


# In[28]:


np.sum(pca.explained_variance_ratio_)


# In[30]:


pca_30 = PCA(n_components=30)
pca_30.fit(scaled_X)


# In[31]:


pca_30.explained_variance_ratio_


# In[32]:


np.sum(pca_30.explained_variance_ratio_)


# In[ ]:





# In[33]:


explained_variance = []

for n in range(1,30):
    pca = PCA(n_components=n)
    pca.fit(scaled_X)
    
    explained_variance.append(np.sum(pca.explained_variance_ratio_))


# In[ ]:





# In[34]:


plt.plot(range(1,30),explained_variance)
plt.xlabel("Number of Components")
plt.ylabel("Variance Explained");


# In[ ]:




