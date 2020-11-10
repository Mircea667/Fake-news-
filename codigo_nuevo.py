# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 11:31:25 2020

@author: PORTATIL
"""

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from nltk.corpus import stopwords


#Combinar csvs

isot_true=pd.read_csv("C:\\Users\PORTATIL\Desktop\TFG\dataset_tfg\isot_dataset\True.csv")
isot_fake=pd.read_csv("C:\\Users\PORTATIL\Desktop\TFG\dataset_tfg\isot_dataset\Fake.csv")
append_TF=isot_true.append(isot_fake,ignore_index=True)

#definicion del corpus

X= append_TF['body'].astype(str) +append_TF['title'].astype(str)
#X_list_in=X.tolist()
#X_2 = [x for x in X if x != 'NaN']
#append_list=X.values.tolist()
y =append_TF['Category']
#y_list=y.tolist()

# Dividir los datos 70% train 30% test
X_train, X_test, y_train, y_test =train_test_split(X, y, train_size=0.70,test_size=0.30)

#Aplicar TFIDF.
#tfIdfVectorizer=TfidfVectorizer(stop_words='english',tokenizer=None,analyzer ='word',token_pattern='(?u)\b\w\w+\b',smooth_idf=True,max_features=100,use_idf=True)
tfIdfVectorizer=TfidfVectorizer()

tfIdf = tfIdfVectorizer.fit_transform(X)

df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF',ascending=False)
print (df.head(25))










