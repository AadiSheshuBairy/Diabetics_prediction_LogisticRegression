#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import seaborn as sns


# In[2]:


df = pd.read_csv("C:\\Users\\Shivanand\\Downloads\\archive (1).zip")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.isnull().sum()
#there are no null values in the data,then we can move to do further steps


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


sns.pairplot(df, diag_kind='kde', hue='Outcome'); 

# Looking at a high level overview of the data separated out by outcome, i.e. 1 = diabetes and 0 = no diabetes
# Lets look at number of patients with diabetes and some of these visuals in more detail


# In[10]:


df['Outcome'].value_counts()
#where outcome : "0" repressent non diabetic and "1" repressent dibetic


# In[11]:


base_color = sns.color_palette()[0]
sns.countplot(data = df, x = 'Outcome', color = base_color);
#where outcome : "0" repressent non diabetic and "1" repressent dibetic


# In[12]:


plt.figure(figsize = [10, 10])
sns.heatmap(df.corr(), annot = True, fmt = '.3f', cmap = 'vlag_r', center = 0);
# Returns a heatmap with Pearson correlation values
# Some interesting correlations including age and number of pregencies


# In[13]:


x = df.iloc[:,[0,1,2,4,5,6,7]].values
y = df.iloc[:,[8]].values
#Preparing the data


# In[14]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
#spliting the data to train and test


# In[15]:


logR = LogisticRegression()
logR.fit(x_train,y_train)
#Training the algorithm


# In[16]:


y_pred = logR.predict(x_test)
# Making Predictions 


# In[17]:


print(accuracy_score(y_pred,y_test))


# In[18]:


print(classification_report(y_pred,y_test))
# Evaluating the algorithm


# In[ ]:




