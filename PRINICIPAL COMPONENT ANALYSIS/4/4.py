#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pylab as pl
from itertools import cycle

iris = load_iris()

numSamples, numFeatures = iris.data.shape
print(numSamples)
print(numFeatures)
print(list(iris.target_names))


# In[ ]:





# In[2]:


X = iris.data
pca = PCA(n_components=2, whiten=True).fit(X)
X_pca = pca.transform(X)


# In[ ]:





# In[3]:


print(pca.components_)


# In[ ]:





# In[4]:


print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))


# In[ ]:





# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *

colors = cycle('rgb')
target_ids = range(len(iris.target_names))
pl.figure()
for i, c, label in zip(target_ids, colors, iris.target_names):
    pl.scatter(X_pca[iris.target == i, 0], X_pca[iris.target == i, 1],
        c=c, label=label)
pl.legend()
pl.show()
    


# In[ ]:




