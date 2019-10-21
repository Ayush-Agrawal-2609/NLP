
# coding: utf-8

# <h1><b> CREATING THE CLASS CONTAINING ALL THE CLEANING FUNCTIONS </b></h1>

# In[ ]:


class nlp_2 :
    
    def __init__(self,n):
        self.n = n
        
    def Cleaning_all(self,text):
        corpus22=[]
            # GENERAL CLEANING
        import nltk
        from nltk.tokenize import sent_tokenize
        import string   
        from nltk.tokenize import word_tokenize
        word_list =  word_tokenize(text)
        from nltk.corpus import stopwords

        #Function to remove stopwords
    
        lang_stopwords = stopwords.words('english')
        stopwords_removed = [w.replace('.','') for w in word_list if w.lower() not in lang_stopwords]
    
        #    " ".join(stopwords_removed)
        
        punt_removed = [w for w in stopwords_removed  if w.lower() not in string.punctuation]
        var1=" ".join(punt_removed)
        var1 = [var1]
        from nltk import PorterStemmer, LancasterStemmer, SnowballStemmer
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        
        #Snowball Stemmer
        
        ss = SnowballStemmer(language='english')
        for w in var1:
            snow = ss.stem(w)
            lem = lemmatizer.lemmatize(snow)
            corpus22.append(lem)
        return corpus22


# <h1><b> CREATING THE DATAFRAME </b></h1>
# 

# In[2]:


import pandas as pd
import os
os.getcwd()
os.chdir('D:\study\ml\others')
os.getcwd()


# In[3]:


df = pd.read_csv('trustpilot_reviews.csv')
df


# <h1><b> CALLING THE FUNCTION </b></h1>
# 

# In[4]:


a = nlp_2(1)
import nltk
nltk.download('wordnet')
corpus1 = []
for i in range(len(df.index)):
    text = df.iloc[i,1]
    b=a.Cleaning_all(text)
    import re
    review = re.sub('[^a-zA-Z]',' ',str(b))
    corpus1.append(review)
print(corpus1)


# <h1><b> TF-IDF </b></h1>
# 

# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
doc_vec1 = vectorizer.fit_transform(corpus1).toarray()
print(doc_vec1)


# In[6]:


import numpy as np
x=doc_vec1
y = df.iloc[:,-1].values


# In[7]:


from sklearn import model_selection
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)  #it splits accordingly without any selection
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report ,accuracy_score
from sklearn import model_selection,neighbors
from sklearn.neighbors import KNeighborsClassifier
model_knn= KNeighborsClassifier()
model_knn.fit(x_train,y_train)
model_knn.score(x_test,y_test)


# In[8]:


a22 = nlp_2(1)
corpus3 = []

#for i in range(len(df.index)):

text = 'I have remitted money 4 days back, but the money is not yet credited in my account in India. Upon inquiry, staff did not have any clue about the reason and asked me to wait until they receive a response from their Head Office. I sent money to pay off my loan installment and delay caused me paying additional amount for late charges and interest.'
b=a22.Cleaning_all(text)
review1 = re.sub('[^a-zA-Z]',' ',str(b))
corpus3.append(review1)
print(corpus3)
y_pred1 = model_knn.predict(vectorizer.transform(corpus3).toarray())


# In[9]:


y_pred1

