#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[ ]:


url="Indian_movies.csv"
df=pd.read_csv(url)


# In[4]:


df.head(5)


# In[5]:


df.dtypes


# In[6]:


df.describe()


# In[7]:


df.describe(include='all')


# In[8]:


df.info()


# In[9]:


import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# # Data filtering and preprocessing

# In[10]:


print(df.columns)


# In[11]:


#check the null value in the dataset
df.isnull().sum()


# In[12]:


shape=df.shape
print(f"The number of rows are {shape[0]}   ,  The number of columns are {shape[1]}")


# In[14]:


unique_generes = df['Genere'].unique()
print(unique_generes)


# In[15]:


rating =df['Rating'].value_counts()
print(rating)


# In[16]:


top_rated_movies=df.sort_values(by ='Rating',ascending=False).head(10)
print(top_rated_movies)


# In[17]:


movie_name_rating=df[['Movie Names','Rating','Genere']]
print(movie_name_rating.head())
plt.figure(figsize=(10,10))
plt.barh(top_rated_movies['Movie Names'],top_rated_movies['Rating'],color='lightpink')
plt.xlabel('Rating')
plt.ylabel('Movie')
plt.title("Top 10 highest -rated Movies")
plt.gca().invert_yaxis()
plt.show()


# In[18]:


df


# In[19]:


columns_of_interest=["Rating","Year","Duration_of_movie"]
sns.set(style='ticks')
sns.pairplot(df[columns_of_interest],diag_kind='kde',markers='o',palette='virdis',height=2.5,aspect=1.2)
plt.subtitle("Rating,year",y=1.02)
plt.show()


# In[21]:


numerical_columns=["Rating","Year","Duration_of_movie"]
correlation_matrix=df[numerical_columns].corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',vmin=-1,vmax=1)
plt.title("correlation heatmap")
plt.show()


# # Feature engineering

# In[22]:


dataset_sorted=df.sort_values(by='Rating',ascending=False)
dataset_sorted['rate_count_persentile']=dataset_sorted['Rating'].rank(pct=True)*100
dataset_sorted.reset_index(drop=True,inplace=True)
print(dataset_sorted[['Movie Names','rate_count_persentile']])


# In[23]:


dataset_sorted.head()


# # Model training and testing

# In[43]:


dataset_sorted['Duration_of_movie'].replace('Not Rated', np.nan, inplace=True)
dataset_sorted


# In[61]:


# Find the minimum non-null value


# Replace NaN values with the minimum value
df['Duration_of_movie'].fillna('2h', inplace=True)


# In[62]:


df['Year']=df['Year'].astype(str)
df['Duration_of_movie']=df['Duration_of_movie'].astype(str)
df['Year']=df['Year'].str.extract('(\d+)').astype(float)
df['Duration_of_movie']=df['Duration_of_movie'].str.extract('(\d+)').astype(float)
x=df[['Year','Duration_of_movie']]
y=df[['Rating']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[53]:


model=LinearRegression()


# In[ ]:





# In[ ]:





# In[37]:


print(min_duration)
df.head(100)


# In[54]:


model.fit(x_train,y_train)


# In[ ]:





# In[ ]:




