#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Libraries 

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


from sklearn.externals import joblib      # For saving model[Dumping] 


# In[ ]:





# In[ ]:


# Mounting Google Drive 
from google.colab import drive
drive.mount('/content/drive/')


# In[ ]:


# Loading Dataset

# 1. Training Data
dataset_train = pd.read_csv('/content/drive/My Drive/Datasets/Large movie review dataset/Train/merged_train.csv')
# 2. Testing Data
dataset_test = pd.read_csv('/content/drive/My Drive/Datasets/Large movie review dataset/Test/merged_test.csv')


# In[ ]:





# In[ ]:


print(dataset_train)


# In[ ]:


print(dataset_test)


# In[ ]:





# In[ ]:


# Checking null values in dataset
print('--- Training Dataset ---')
print(dataset_train.isnull().sum())
print('')

print('--- Testing Dataset ---')
print(dataset_test.isnull().sum())


# In[ ]:





# # Label Encoding

# In[ ]:


# Label Encoding
le = LabelEncoder()

dataset_train['sentiment'] = le.fit_transform(dataset_train['sentiment'])
dataset_test['sentiment'] = le.fit_transform(dataset_test['sentiment'])


# In[ ]:





# # Analysing dataset after label encoding
# 

# In[ ]:


# Positive - 1
# Negative - 0
print(dataset_train)


# In[ ]:


print(dataset_test)


# In[ ]:





# In[ ]:


# Seprating dataset into X and Y
y = dataset_train['sentiment']
x = dataset_train['comment']


# In[ ]:





# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(ngram_range=(1,2))


# In[ ]:





# In[ ]:


x = cv.fit_transform(x)


# In[ ]:





# In[ ]:


LRModel = LogisticRegression()


# In[ ]:





# In[ ]:


LRModel.fit(x,y)


# In[ ]:


X = dataset_test['comment']
Y = dataset_test['sentiment']

X, Y


# In[ ]:





# In[ ]:


pred_y = LRModel.predict(cv.transform(X))


# In[ ]:


score = accuracy_score(Y,pred_y)


# In[ ]:


score


# In[ ]:





# In[ ]:


joblib.dump(LRModel , '/content/drive/My Drive/Datasets/SentimentAnalysisModel.ds1')


# In[ ]:





# In[ ]:




