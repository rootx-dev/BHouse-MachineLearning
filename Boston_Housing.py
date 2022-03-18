#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np


# ## Load Boston Houses Dataset
# 

# In[2]:


(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.boston_housing.load_data()


# ## Check shapes and types
# 

# In[3]:


type(train_data), train_data.dtype, train_data.shape


# In[4]:


type(train_labels), train_labels.dtype, train_labels.shape


# In[5]:


train_data.shape


# In[6]:


train_data[2], train_labels[2]


# In[7]:


test_data.shape


# In[ ]:





# ## reshape to (-1,1)

# In[8]:


train_labels = np.reshape(train_labels, (-1,1))
test_labels = np.reshape(test_labels, (-1,1))


# In[9]:


train_labels.shape, test_labels.shape


# ## check min and max of data

# In[10]:


train_data.min(axis=0), train_data.max(axis=0)


# # Normalize the data

# ## using MinMax Scaler from sklearn

# ## fit and then transform

# In[11]:


from sklearn.preprocessing import MinMaxScaler


# In[12]:


my_mmscaler = MinMaxScaler()


# In[13]:


my_mmscaler.fit(train_data)


# In[14]:


train_data = my_mmscaler.transform(train_data)


# In[ ]:





# ## check min and max of data after normalization

# In[15]:


train_data.min(axis=0), train_data.max(axis=0)


# ## transfer test data as well

# ## and check min and max of test data

# In[16]:


test_data = my_mmscaler.transform(test_data)


# In[17]:


test_data.min(axis=0), test_data.max(axis=0)


# In[18]:


test_labels.min(), test_labels.max(), train_labels.min(), train_data.max()


# In[19]:


##train_labels = train_labels / 50.
##test_labels = test_labels / 50.


# In[20]:


test_labels.min(), test_labels.max(), train_labels.min(), train_labels.max()


# # Create the regression model

# ## import layers and model

# In[21]:


from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential


# ## create the model

# In[28]:


input_layer = Input((13,))
hidden_layer =  Dense(8, activation='relu')(input_layer)
hidden_layer =  Dense(4, activation='relu')(hidden_layer)

prediction = Dense(1)(hidden_layer)
first_regression_model = Model(input_layer, prediction)


# ## print summary of the model

# In[29]:


first_regression_model.summary()


# ## compile the model with optimizer and loss function
# 

# In[30]:


from tensorflow.keras.optimizers import SGD, Adam


# In[31]:


first_regression_model.compile(optimizer='adam', loss='mse')


# ## Now we can train the model

# In[32]:


first_regression_model.fit(x=train_data, y=train_labels, batch_size=8, 
                           epochs=500, validation_data=(test_data, test_labels))


# In[33]:


model_history = pd.DataFrame(first_regression_model.history.history)


# In[34]:


model_history.plot()


# ## Evaluate the model

# In[35]:


model_predictions = first_regression_model.predict(test_data, batch_size=8)


# In[36]:


model_predictions[1], test_labels[1]


# In[37]:


from tensorflow.python.platform.tf_logging import error
errors =0.0
for index in range(len(test_labels)):
  if abs(model_predictions[index] - test_labels[index]) > 5.0:
    errors += 1.

print(1. - (errors/ len(test_labels)))


# In[ ]:




