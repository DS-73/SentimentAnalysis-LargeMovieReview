#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import os
import string
import pandas as pd
from nltk.corpus import stopwords


# In[ ]:





# In[2]:


dirpath = "C:\\Users\\DS\\Desktop\\MLminiProject\\aclImdb\\train\\neg\\"
output = "C:\\Users\\DS\\Desktop\\MLminiProject\\aclImdb\\CSV\\Train\\"
# csvout = pd.DataFrame()
files = os.listdir(dirpath)


# In[ ]:





# In[3]:


stop_words= ['/', '<', '>', '<br>', '<b>', '<u>', '<i>', '<html>', '<body>', '</br>', '</b>', '</u>', '</i>', 
             '</html>', '</body>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# In[ ]:





# In[4]:


# Replacing Non-ASCII Character with space


# In[5]:


def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]',' ', text)


# In[ ]:





# In[6]:


# Training dataset


# In[ ]:





# In[7]:


# Train - Negative CSV creating
# Giving header to CSV column

FileName = open(output+"neg_train.csv", "a")
FileName.write("comment,sentiment\n")
FileName.close()

counter = 1
for filename in files:
    text = open(dirpath+filename, errors="ignore").read()
    text = remove_non_ascii(text)
    lower_text = text.lower()
    punctuations_text = lower_text.translate(str.maketrans("", "", string.punctuation))
    
    FileName = open(output+"neg_train.csv", "a")
    FileName.write("%s,negative\n"%(punctuations_text))
    FileName.close()
    
    print("Successfully Compiled << ", counter, " : ", filename," >>")
    counter = counter + 1
    
print("Done ... ")


# In[ ]:





# In[ ]:





# In[8]:


dirpath = "C:\\Users\\DS\\Desktop\\MLminiProject\\aclImdb\\train\\pos\\"
output = "C:\\Users\\DS\\Desktop\\MLminiProject\\aclImdb\\CSV\\Train\\"

files = os.listdir(dirpath)
#print(files)


# In[ ]:





# In[9]:


# Train - Positive CSV creating
# Giving header to CSV column

FileName = open(output+"pos_train.csv", "a")
FileName.write("comment,sentiment\n")
FileName.close()

counter = 1
for filename in files:
    text = open(dirpath+filename, errors="ignore").read()
    text = remove_non_ascii(text)
    lower_text = text.lower()
    punctuations_text = lower_text.translate(str.maketrans("", "", string.punctuation))
    
    FileName = open(output+"pos_train.csv", "a")
    FileName.write("%s,positive\n"%(punctuations_text))
    FileName.close()
    
    print("Successfully Compiled << ", counter, " : ", filename," >>")
    counter = counter + 1
    
print("Done ... ")


# In[ ]:





# In[ ]:





# In[10]:


# Testing Dataset


# In[ ]:





# In[11]:


dirpath = "C:\\Users\\DS\\Desktop\\MLminiProject\\aclImdb\\test\\neg\\"
output = "C:\\Users\\DS\\Desktop\\MLminiProject\\aclImdb\\CSV\\Test\\"
# csvout = pd.DataFrame()
files = os.listdir(dirpath)


# In[ ]:





# In[12]:


# Test - Negative CSV creating
# Giving header to CSV column

FileName = open(output+"neg_test.csv", "a")
FileName.write("comment,sentiment\n")
FileName.close()

counter = 1
for filename in files:
    text = open(dirpath+filename, errors="ignore").read()
    text = remove_non_ascii(text)
    lower_text = text.lower()
    punctuations_text = lower_text.translate(str.maketrans("", "", string.punctuation))
    
    FileName = open(output+"neg_test.csv", "a")
    FileName.write("%s,negative\n"%(punctuations_text))
    FileName.close()
    
    print("Successfully Compiled << ", counter, " : ", filename," >>")
    counter = counter + 1
    
print("Done ... ")


# In[ ]:





# In[13]:


dirpath = "C:\\Users\\DS\\Desktop\\MLminiProject\\aclImdb\\test\\pos\\"
output = "C:\\Users\\DS\\Desktop\\MLminiProject\\aclImdb\\CSV\\Test\\"
# csvout = pd.DataFrame()
files = os.listdir(dirpath)


# In[ ]:





# In[14]:


# Train - Positive CSV creating
# Giving header to CSV column

FileName = open(output+"pos_test.csv", "a")
FileName.write("comment,sentiment\n")
FileName.close()


counter = 1
for filename in files:
    text = open(dirpath+filename, errors="ignore").read()
    text = remove_non_ascii(text)
    lower_text = text.lower()
    punctuations_text = lower_text.translate(str.maketrans("", "", string.punctuation))
    
    FileName = open(output+"pos_test.csv", "a")
    FileName.write("%s,positive\n"%(punctuations_text))
    FileName.close()
    
    print("Successfully Compiled << ", counter, " : ", filename," >>")
    counter = counter + 1
    
print("Done ... ")


# In[ ]:





# In[15]:


Negative = pd.read_csv(output + "neg_test.csv")


# In[16]:


Negative.head()


# In[17]:


Negative.info()


# In[ ]:





# In[ ]:





# In[ ]:




