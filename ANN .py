#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset = pd.read_csv("Churn_Modelling.csv")


# In[3]:


dataset.columns


# In[4]:


y = dataset['Exited']
X = dataset[['CreditScore',
       'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary']]


# In[5]:


X


# In[6]:


geo = dataset['Geography']


# In[7]:


geo = pd.get_dummies(geo , drop_first=True)


# In[8]:


gender = dataset['Gender']


# In[9]:


gender = pd.get_dummies(gender , drop_first=True)


# In[10]:


X


# In[11]:


X = pd.concat([X, geo ,gender],axis=1)


# In[12]:


X.isnull()


# In[13]:


X


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)


# In[37]:


from keras.models import Sequential


# In[38]:


import seaborn as sns


# In[39]:


X.columns


# In[40]:


age = X['Age']


# In[41]:


sns.scatterplot(X)


# In[42]:


#import matplotlib.pyplot as plt


# In[43]:


#plt.scatter(age , y)


# In[44]:


model = Sequential()


# In[45]:


from keras.layers import Dense


# In[46]:


X_train 


# In[47]:


y_train


# In[48]:


model.add(Dense()) # error because they need units 


# In[49]:


model.add(Dense(units=6 , input_dim=11))


# In[50]:


model.add(Dense(units=6))
model.add(Dense(units=8))
model.add(Dense(units=1 , activation='sigmoid'))


# In[51]:


#model.compile() #they need optimizer 


# In[52]:


model.compile(optimizer=Adam()) #fail because we need to import adam and also set loss on it 


# In[53]:


from keras.optimizers import Adam


# In[56]:


model.compile(optimizer=Adam(),loss='binary_crossentropy',metrics=['accuracy'])


# In[57]:


model.fit(X_train , y_train , epochs=50)


# In[30]:


model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy')


# In[31]:


#first layer is input layer then 2 layer is hidden layer 
model.add(Dense(units=6, input_dim=11 , activation='relu'))
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=8, activation='relu'))


# In[32]:


#output layer
model.add(Dense(units=1 , activation='sigmoid')


# In[33]:


model.compile(optimizer=Adam(learning_rate=0.000001),loss='binary_crossentropy' )


# In[34]:


model.fit(X_train,y_train , epochs=200 , verbose=0)


# In[35]:


df_loss = pd.DataFrame(model.history.history)


# In[36]:


df_loss.plot()


# In[ ]:




