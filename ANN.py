#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset = pd.read_csv('weight-height.csv')


# In[3]:


dataset.info()


# In[4]:


dataset.columns


# In[5]:


y = dataset['Weight']


# In[6]:


X = dataset['Height']


# In[7]:


from keras.models import Sequential


# In[8]:


from keras.layers import Dense


# In[9]:


from keras.optimizers import Adam


# In[10]:


model = Sequential()


# In[11]:


# by default : linear activation functions
# units : output
# input shape: input feature
# dense: how many hidden layer
model.add(Dense(units=1 , input_shape=(1,)  ))


# In[12]:


model.summary()


# In[18]:


model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.000001) , metrics=['accuracy'])


# In[19]:


model.fit(X,y, epochs=20)


# In[20]:


W , B = model.get_weights()


# In[21]:


W


# In[22]:


B


# In[23]:


W.shape


# In[24]:


W[0,0] = 0.0


# In[25]:


B[0] = 0.0


# In[26]:


model.set_weights((W,B))


# In[27]:


model.get_weights()


# In[ ]:




