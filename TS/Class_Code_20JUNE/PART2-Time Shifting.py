#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[8]:


df = pd.read_csv(r'G:\Traning\Data/starbucks.csv', index_col='Date',parse_dates=True)
#df = pd.read_csv(r'G:\Traning\Data/starbucks.csv')


# In[9]:


df.head()


# In[10]:


df.tail()


# In[11]:


#
df.shift(1).head()


# In[12]:


df.shift(1).tail()


# In[13]:


df.shift(-1).head()


# In[14]:


#shifting based on time series fre code


# In[15]:


# shift everyting forward one month
df.shift(periods=1, freq='M').head()


# In[ ]:




