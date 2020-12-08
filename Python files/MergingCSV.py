#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd


# In[ ]:





# In[2]:


dirpath = "C:\\Users\\DS\\Desktop\\MLminiProject\\aclImdb\\CSV\\Train\\"
output = "C:\\Users\\DS\\Desktop\\MLminiProject\\aclImdb\\CSV\\Test\\"


# In[ ]:





# # Training CSV

# In[3]:


train_neg = pd.read_csv(dirpath + "neg_train.csv")
train_neg


# In[ ]:





# In[4]:


train_pos = pd.read_csv(dirpath + "pos_train.csv")
train_pos


# In[ ]:





# In[5]:


test_merged = pd.concat([train_neg, train_pos])
#test_merged.info()

test_merged.to_csv(dirpath + "merged_train.csv", index = False, header=True)
test_merged = pd.read_csv(dirpath + "merged_train.csv")
test_merged.info()


# In[ ]:





# In[ ]:





# # Testing CSV

# In[6]:


test_pos = pd.read_csv(output + "pos_test.csv")
test_pos


# In[ ]:





# In[7]:


test_neg = pd.read_csv(output + "neg_test.csv")
test_neg


# In[ ]:





# In[8]:


test_merged = pd.concat([test_neg, test_pos])
#test_merged.info()

test_merged.to_csv(output+"merged_test.csv", index = False, header=True)
test_merged = pd.read_csv(output + "merged_test.csv")
test_merged.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




