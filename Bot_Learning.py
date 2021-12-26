#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras .optimizers import Adam
import pandas as pd
import numpy as np
import numpy_indexed as npi


# In[2]:


symp_data= pd.read_csv('dataset.csv')


# In[3]:


symptoms= symp_data[['Symptom_1','Symptom_2','Symptom_3','Symptom_4','Symptom_5','Symptom_6','Symptom_7','Symptom_8','Symptom_9','Symptom_10','Symptom_11','Symptom_12','Symptom_13','Symptom_14','Symptom_15','Symptom_16','Symptom_17']]
Disease= symp_data[['Disease']]
symptoms['Symptom_1'].tail()
symptoms


# In[4]:


df= pd.read_csv('Symptom-severity.csv')
df
for x in df['Symptom']:
    print(x)


# In[5]:


df.drop('weight', axis=1)
symp_dict= df['Symptom'].to_dict()
symp_dict
new_dict = dict([(value, key) for key, value in symp_dict.items()])
new_dict['dischromic _patches']=102



# In[6]:


symptoms['Symptom_1']= symptoms['Symptom_1'].str.strip()
symptoms['Symptom_1']= symptoms['Symptom_1'].map(new_dict)
symptoms['Symptom_2']= symptoms['Symptom_2'].str.strip()
symptoms['Symptom_2']= symptoms['Symptom_2'].map(new_dict)
symptoms['Symptom_3']= symptoms['Symptom_3'].str.strip()
symptoms['Symptom_3']= symptoms['Symptom_3'].map(new_dict)
symptoms['Symptom_4']= symptoms['Symptom_4'].str.strip()
symptoms['Symptom_4']= symptoms['Symptom_4'].map(new_dict)
symptoms['Symptom_5']= symptoms['Symptom_5'].str.strip()
symptoms['Symptom_5']= symptoms['Symptom_5'].map(new_dict)
symptoms['Symptom_6']= symptoms['Symptom_6'].str.strip()
symptoms['Symptom_6']= symptoms['Symptom_6'].map(new_dict)
symptoms['Symptom_7']= symptoms['Symptom_7'].str.strip()
symptoms['Symptom_7']= symptoms['Symptom_7'].map(new_dict)
symptoms['Symptom_8']= symptoms['Symptom_8'].str.strip()
symptoms['Symptom_8']= symptoms['Symptom_8'].map(new_dict)
symptoms['Symptom_9']= symptoms['Symptom_9'].str.strip()
symptoms['Symptom_9']= symptoms['Symptom_9'].map(new_dict)
symptoms['Symptom_10']= symptoms['Symptom_10'].str.strip()
symptoms['Symptom_10']= symptoms['Symptom_10'].map(new_dict)
symptoms['Symptom_11']= symptoms['Symptom_11'].str.strip()
symptoms['Symptom_11']= symptoms['Symptom_11'].map(new_dict)
symptoms['Symptom_12']= symptoms['Symptom_12'].str.strip()
symptoms['Symptom_12']= symptoms['Symptom_12'].map(new_dict)
symptoms['Symptom_13']= symptoms['Symptom_13'].str.strip()
symptoms['Symptom_13']= symptoms['Symptom_13'].map(new_dict)
symptoms['Symptom_14']= symptoms['Symptom_14'].str.strip()
symptoms['Symptom_14']= symptoms['Symptom_14'].map(new_dict)
symptoms['Symptom_15']= symptoms['Symptom_15'].str.strip()
symptoms['Symptom_15']= symptoms['Symptom_15'].map(new_dict)
symptoms['Symptom_16']= symptoms['Symptom_16'].str.strip()
symptoms['Symptom_16']= symptoms['Symptom_16'].map(new_dict)
symptoms['Symptom_17']= symptoms['Symptom_17'].str.strip()
symptoms['Symptom_17']= symptoms['Symptom_17'].map(new_dict)



# In[ ]:





# In[7]:


symptoms.fillna(133, inplace=True)


# In[8]:


symptoms







