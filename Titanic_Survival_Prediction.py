#!/usr/bin/env python
# coding: utf-8

# # Importing the dependencies

# In[144]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # Data collection and preprocessing

# In[145]:


#load the data from csv file to a pandas dataframe
titanic_data=pd.read_csv('train.csv')
titanic_data.head()


# In[65]:


titanic_data.shape


# In[146]:


#information about the data
titanic_data.info()


# In[67]:


titanic_data.isnull().sum()


# In[147]:


titanic_data=titanic_data.drop(columns = 'Cabin',axis=1)


# In[69]:


titanic_data.head()


# In[149]:


titanic_data.shape


# In[71]:


#information about the data
titanic_data.info()


# In[150]:


#check the number of missing values in each columns
titanic_data.isnull().sum()


# In[74]:


#Replace missing value(Age) with the mean value of age
titanic_data.fillna(titanic_data['Age'].mean(),inplace=True)


# In[151]:


#finding the mode value of the embarked columns
print(titanic_data['Embarked'].mode())


# In[76]:


print(titanic_data['Embarked'].mode()[0])


# In[78]:


#replacing missing value(Embarked) with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True)


# In[80]:


titanic_data.info()


# # Data Analysis 

# In[81]:


#Getting some stastical measures about the data
titanic_data.describe()


# In[82]:


#finding the number of people survived or not survived
titanic_data['Survived'].value_counts()


# In[83]:


#Data visualization


# In[84]:


sns.set()


# In[90]:


#Making a count plot for 'survived' columns
# Create a count plot
sns.countplot(x='Survived', data=titanic_data)

# Add labels and title
plt.xlabel('Survived')
plt.ylabel('Count')
plt.title('Count of Survived Passengers in Titanic')

# Show the plot
plt.show()


# In[91]:


#Number of survivers Gender wise
sns.countplot(x='Sex',hue='Survived',data=titanic_data)


# In[93]:


sns.countplot(x='Sex',hue = 'Pclass',data=titanic_data)


# In[94]:


#Encoding the categorical columns
titanic_data['Sex'].value_counts()


# In[95]:


titanic_data['Embarked'].value_counts()


# In[98]:


#converting categorical columns 
titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)


# In[99]:


titanic_data.head()


# In[101]:


#Separating feaatures and target
x=titanic_data.drop(columns=['PassengerId','Name','Ticket'],axis=1)
y=titanic_data['Survived']


# In[102]:


print(x)


# In[103]:


print(y)


# In[104]:


# Splitting the data into training data and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[106]:


print(x.shape,x_train.shape,x_test.shape)


# In[121]:


#LogisticRegression model
model=LogisticRegression()


# In[134]:


#training the logistic regression model with training data
model.fit(x_train, y_train)


# # model evaluation
# # Accuracy score 
# 
# 

# In[135]:


# Accuracy on training data
x_train_prediction=model.predict(x_train)


# In[136]:


print(x_train_prediction)


# In[138]:


training_data_accuracy=accuracy_score(y_train,x_train_prediction)
print("Accuracy score of training data",training_data_accuracy)


# In[139]:


#acccuracy on test data
x_test_prediction=model.predict(x_test)


# In[140]:


print(x_test_prediction)


# In[142]:


# Accuracy on test data
x_test_prediction=model.predict(x_test)


# In[143]:


test_data_accuracy=accuracy_score(y_test,x_test_prediction)
print("Accuracy score of test data:",test_data_accuracy)


# In[ ]:




