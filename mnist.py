#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist


# In[2]:


dataset = mnist.load_data('mymnist.db')


# In[3]:


len(dataset)


# In[4]:


train , test = dataset


# In[5]:


len(train)


# In[6]:


X_train , y_train = train


# In[7]:


X_train.shape


# In[8]:


X_test , y_test = test


# In[9]:


X_test.shape


# In[10]:


img1 = X_train[7]


# In[11]:


img1.shape


# In[12]:


img1_label = y_train[7]


# In[13]:


img1_label


# In[14]:


img1.shape


# In[15]:


img1.shape


# In[16]:


img1_1d = img1.reshape(28*28)


# In[17]:


img1_1d.shape


# In[18]:


X_train.shape


# In[19]:


X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)


# In[20]:


X_train_1d.shape


# In[21]:


X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')


# In[22]:


X_train.shape


# In[23]:


y_train.shape


# In[24]:


from keras.utils.np_utils import to_categorical


# In[25]:


y_train_cat = to_categorical(y_train)


# In[26]:


y_train_cat


# In[27]:


y_train_cat[7]


# In[28]:


from keras.models import Sequential


# In[29]:


from keras.layers import Dense


# In[30]:


model = Sequential()


# In[31]:


model.add(Dense(units=512, input_dim=28*28, activation='relu'))


# In[32]:


model.summary()


# In[33]:


model.add(Dense(units=256, activation='relu'))


# In[34]:


model.add(Dense(units=128, activation='relu'))


# In[35]:


model.add(Dense(units=32, activation='relu'))


# In[36]:


model.summary()


# In[37]:


model.add(Dense(units=10, activation='softmax'))


# In[38]:


model.summary()


# In[39]:


from keras.optimizers import RMSprop


# In[40]:


model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )


# In[41]:


history = model.fit(X_train, y_train_cat, epochs=1)


# In[45]:


print(max(history.history['accuracy']))
if (max(history.history['accuracy'])) > .80 :
    model.save('model.h5')


# In[47]:


accuracy = open('/root/task31/accuracy.txt','w+')
accuracy.write (str(history.history['accuracy']))
accuracy.close()


# In[ ]:




