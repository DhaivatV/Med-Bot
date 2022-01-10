#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from joblib import dump, load


# In[2]:


symptoms= pd.read_csv('cleansed_symp_data')
diseases= pd.read_csv('Disease_data')


# In[3]:


diseases.tail()


# In[4]:


symptoms.head()


# In[5]:


x_train, x_test, y_train, y_test= train_test_split(symptoms, diseases, test_size=0.2, random_state=42)


# In[6]:


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


# In[12]:


filename= 'Med_Bot.model'


# In[13]:


model.save_model(filename)


# In[ ]:





# In[ ]:




