#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\lenovo\Downloads\cancer_detection\data.csv")
df


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


df.tail()


# In[10]:


df.isnull().sum()


# In[11]:


df['radius_mean'].value_counts()


# In[12]:


df['diagnosis'].value_counts()


# In[13]:


df['texture_mean'].value_counts()


# In[14]:


df['perimeter_mean'].value_counts()


# In[15]:


df['area_mean'].value_counts()


# In[16]:


df['area_worst'].value_counts()


# In[17]:


df=df.dropna(axis=1)


# In[19]:


df


# In[20]:


sns.countplot(df['diagnosis'],label="count")


# In[21]:


from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()
df.iloc[:,1]=labelencoder_Y.fit_transform(df.iloc[:,1].values)


# In[22]:


df.head()


# In[24]:


sns.pairplot(df.iloc[:,1:5],hue='diagnosis')


# In[25]:


df.iloc[:,1:32].corr()


# In[27]:


plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:32].corr(),annot=True)


# In[28]:


x=df.iloc[:,2:31].values
y=df.iloc[:,1].values


# In[29]:


x


# In[30]:


y


# In[32]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


# In[33]:


# feature scaling
from sklearn.preprocessing import StandardScaler
x_train=StandardScaler().fit_transform(x_train)
x_test=StandardScaler().fit_transform(x_test)


# In[34]:


def models(X_train,Y_train):
        #logistic regression
        from sklearn.linear_model import LogisticRegression
        log=LogisticRegression(random_state=0)
        log.fit(X_train,Y_train)
        
        
        #Decision Tree
        from sklearn.tree import DecisionTreeClassifier
        tree=DecisionTreeClassifier(random_state=0,criterion="entropy")
        tree.fit(X_train,Y_train)
        
        #Random Forest
        from sklearn.ensemble import RandomForestClassifier
        forest=RandomForestClassifier(random_state=0,criterion="entropy",n_estimators=10)
        forest.fit(X_train,Y_train)
        
        print('[0]logistic regression accuracy:',log.score(X_train,Y_train))
        print('[1]Decision tree accuracy:',tree.score(X_train,Y_train))
        print('[2]Random forest accuracy:',forest.score(X_train,Y_train))
        
        return log,tree,forest


# In[36]:


model=models(x_train,y_train)


# In[37]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

for i in range(len(model)):
    print("Model",i)
    print(classification_report(y_test,model[i].predict(x_test)))
    print('Accuracy : ',accuracy_score(y_test,model[i].predict(x_test)))


# In[38]:


pred=model[2].predict(x_test)
print('Predicted values:')
print(pred)
print('Actual values:')
print(y_test)


# In[39]:


from joblib import dump
dump(model[2],"Cancer_prediction.joblib")


# In[ ]:




