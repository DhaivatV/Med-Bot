#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


symptoms= pd.read_csv('cleansed_symp_data')
diseases= pd.read_csv('Disease_data')


# In[3]:


diseases.tail()


# In[4]:


symptoms.head()


# In[ ]:





# In[5]:


x_train, x_test, y_train, y_test= train_test_split(symptoms, diseases, test_size=0.2, random_state=42)


# In[6]:


import xgboost as xgb

train = xgb.DMatrix(x_train, label=y_train)
test = xgb.DMatrix(x_test, label=y_test)


# In[7]:


param = {
    'max_depth': 1,
    'eta': 0.3,
    'objective': 'multi:softmax',
    'num_class' : 401 } 
epochs = 26


# In[8]:


model = xgb.train(param, train, epochs)


# In[9]:


predictions = model.predict(test)


# In[10]:


print(predictions)


# In[11]:


accu= ((accuracy_score(y_test, predictions))*100)
print(f'Accuracy:{accu} %')


# In[ ]:





# In[ ]:




