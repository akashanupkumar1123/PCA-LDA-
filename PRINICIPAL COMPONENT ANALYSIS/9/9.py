#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[3]:


digits = pd.read_csv('digits.csv')


# In[4]:


digits


# In[6]:


pixels = digits.drop('number_label',axis=1)


# In[7]:


pixels


# In[ ]:





# In[9]:


single_image = pixels.iloc[0]


# In[10]:


single_image


# In[ ]:





# In[11]:


single_image.to_numpy()


# In[ ]:





# In[12]:


single_image.to_numpy().shape


# In[ ]:





# In[13]:


single_image.to_numpy().reshape(8,8)


# In[ ]:





# In[14]:


plt.imshow(single_image.to_numpy().reshape(8,8))


# In[ ]:





# In[15]:


plt.imshow(single_image.to_numpy().reshape(8,8),cmap='gray')


# In[ ]:





# In[16]:


sns.heatmap(single_image.to_numpy().reshape(8,8),annot=True,cmap='gray')


# In[ ]:





# In[17]:


from sklearn.preprocessing import StandardScaler


# In[18]:


scaler = StandardScaler()


# In[19]:


scaled_pixels = scaler.fit_transform(pixels)


# In[20]:


scaled_pixels


# In[ ]:





# In[21]:


from sklearn.decomposition import PCA


# In[22]:


pca_model = PCA(n_components=2)


# In[23]:


pca_pixels = pca_model.fit_transform(scaled_pixels)


# In[ ]:





# In[24]:


np.sum(pca_model.explained_variance_ratio_)


# In[ ]:





# In[25]:


plt.figure(figsize=(10,6),dpi=150)
labels = digits['number_label'].values
sns.scatterplot(pca_pixels[:,0],pca_pixels[:,1],hue=labels,palette='Set1')
plt.legend(loc=(1.05,0))


# In[26]:


# You should see label #4 as being the most separated group, 
# implying its the most distinct, similar situation for #2, #6 and #9.


# In[27]:


from sklearn.decomposition import PCA


# In[28]:


pca_model = PCA(n_components=3)


# In[29]:


pca_pixels = pca_model.fit_transform(scaled_pixels)


# In[30]:


from mpl_toolkits import mplot3d


# In[32]:


plt.figure(figsize=(8,8),dpi=150)
ax = plt.axes(projection='3d')
ax.scatter3D(pca_pixels[:,0],pca_pixels[:,1],pca_pixels[:,2],c=digits['number_label']);


# In[ ]:





# In[33]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[34]:


ax = plt.axes(projection='3d')
ax.scatter3D(pca_pixels[:,0],pca_pixels[:,1],pca_pixels[:,2],c=digits['number_label']);


# In[ ]:




