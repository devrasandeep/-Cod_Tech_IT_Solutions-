#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


#loading the data set to a pandas daaframe
credit_card_data=pd.read_csv('creditcard.csv')


# In[3]:


credit_card_data.head()


# In[4]:


#dataset information
credit_card_data.info()


# In[16]:


#checkinf the number of missing value in the column
Sum=credit_card_data.isnull().sum()
Sum


# # 0 means secure transition/legit 1 means fraud transaction

# In[5]:


credit_card_data['Class'].value_counts()


# In[20]:


# This dataset is highly unbalanced
#  0-> Normal Transaction
#  1->Fraudulent Transaction
# separating the data for analysis


# In[6]:


legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]


# In[7]:


print(legit.shape)
print(fraud.shape)


# In[8]:


legit.Amount.describe()


# In[9]:


fraud.Amount.describe()


# In[10]:


#compare the values for both transactions


# In[11]:


credit_card_data.groupby('Class').mean()


# under-Sampling

# build a sample dataset containing similar distribution of normal transactions and the fraudlent transactions

# Number of fraudlent transactions ->  492

# In[12]:


legit_sample = legit.sample(n=492)


# concatenating two DataFrames

# In[13]:


new_dataset=pd.concat([legit_sample,fraud],axis=0)


# In[14]:


new_dataset.shape


# In[16]:


new_dataset['Class'].value_counts()


# In[17]:


new_dataset.groupby('Class').mean()


# split the data into features and targets

# In[18]:


x=new_dataset.drop(columns='Class',axis=1)
y=new_dataset['Class']


# In[19]:


print(x)


# In[24]:


print(y)


# # split the data into Training Data and Testing Data

# In[25]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
print(x.shape,x_train,x_test)
print(y.shape,y_train,y_test)


# # Model Training 
# # Logistic Regression

# In[35]:


model=LogisticRegression()


# In[36]:


#training logistic regression model with Training Data
model.fit(x_train,y_train)


# # Model Evaluation
# # Accuracy score

# In[38]:


#accuracy on training data


# In[39]:


x_train_prediction =model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)


# In[42]:


print("Accuracy on taining data is ",training_data_accuracy)


# In[43]:


#Accuracy on test data
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)


# In[44]:


print("Accuracy on testing data is",test_data_accuracy)


# In[ ]:




