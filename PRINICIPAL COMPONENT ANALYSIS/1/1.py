#!/usr/bin/env python
# coding: utf-8

# In[14]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


from sklearn.datasets import load_breast_cancer


# In[16]:


cancer = load_breast_cancer()


# In[17]:


cancer.keys()


# In[ ]:





# In[18]:


print(cancer['DESCR'])


# In[ ]:





# In[19]:


df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
#(['DESCR', 'data', 'feature_names', 'target_names', 'target'])


# In[20]:


df.head()


# In[ ]:





# In[21]:


from sklearn.preprocessing import StandardScaler


# In[22]:


scaler = StandardScaler()
scaler.fit(df)


# In[23]:


scaled_data = scaler.transform(df)


# In[ ]:





# In[24]:


from sklearn.decomposition import PCA


# In[25]:


pca = PCA(n_components=2)


# In[26]:


pca.fit(scaled_data)


# In[ ]:





# In[27]:


x_pca = pca.transform(scaled_data)


# In[28]:


scaled_data.shape


# In[29]:


x_pca.shape


# In[ ]:





# In[30]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[ ]:





# In[31]:


pca.components_


# In[32]:


df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])


# In[ ]:





# In[33]:


plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)


# In[ ]:




