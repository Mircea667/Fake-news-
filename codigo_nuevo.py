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
from nltk.corpus import stopwords


#Combinar csvs

isot_true=pd.read_csv("C:\\Users\PORTATIL\Desktop\TFG\dataset_tfg\isot_dataset\True.csv")
isot_fake=pd.read_csv("C:\\Users\PORTATIL\Desktop\TFG\dataset_tfg\isot_dataset\Fake.csv")
append_TF=isot_true.append(isot_fake,ignore_index=True)

#definicion del corpus
append_TF['body']=append_TF['body']
append_TF['title']=append_TF['title']

append_TF2=append_TF['body']+append_TF['title']
append_TF2=append_TF2

append_list=append_TF2.values.tolist()
list_string= [str(x) for x in append_list]
print(list_string)


tfIdfVectorizer=TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(list_string)
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df.head(25))









