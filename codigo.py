# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


#Combinar csvs

isot_true=pd.read_csv("C:\\Users\PORTATIL\Desktop\TFG\dataset_tfg\isot_dataset\True.csv")
isot_fake=pd.read_csv("C:\\Users\PORTATIL\Desktop\TFG\dataset_tfg\isot_dataset\Fake.csv")
append_TF=isot_true.append(isot_fake,ignore_index=True)

#Conversion
print("Esquema \n\n",append_TF.dtypes)
print("Numero de columnas",append_TF.shape)

append_TF['body']=append_TF['body'].apply(str)
append_TF['title']=append_TF['title'].apply(str)

#Preprocesamiento
import re 
def preprocess(text):
    #minúscula
    text=text.lower()
    #quitar etiquetas
    text=re.sub("&lt;/?.*?&gt;","&lt;&gt;",text)
    #quitar caracteres especiales y digitos
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text


append_TF['text']=append_TF['body']+ append_TF['title']
append_TF['text']=append_TF['text'].apply(lambda x:preprocess(x))

nltk.download('stopwords')

def get_stop_words(stop_file_path):
    """load stop words"""
    with open(stop_file_path,'r',encoding='utf-8') as f:
        stopwords=f.readlines()
        stop_set=set(m.strip() for m in stopwords)
        return frozenset(stop_set)
#cargar un conjunto de stopwords
stopwords= set(stopwords.words('english'))

#Columna texto
docs=append_TF['text'].tolist()

#eliminar stopword
#ignorar palabras que aparecen en el 85% de los documentos

cv=CountVectorizer(max_df=0.85,stop_words=stopwords)        
word_count_vector=cv.fit(docs)

list(cv.get_feature_names())[:]
#TF-IDF
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)


# test de cuerpo y titulo de la noticia
df_test1=pd.read_csv("C:\\Users\PORTATIL\Desktop\TFG\dataset_tfg\isot_dataset\Test_True.csv",lines=True)
df_test2=pd.read_csv("C:\\Users\PORTATIL\Desktop\TFG\dataset_tfg\isot_dataset\Test_Fake.csv",lines=True)
df_test3=df_test1.append(df_test2,ignore_index=True)
df_test3['text'] = df_test3['title'] + df_test3['body']
df_test3['text'] =df_test3['text'].apply(lambda x:pre_process(x))


# metemos el cuerpo de la noticia y la cabecera en una lista
docs_test=df_test3['text'].tolist()
docs_title=df_test3['title'].tolist()
docs_body=df_test3['body'].tolist()

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

#TF IDF para todos los documentos
tf_idf_vector=tfidf_transformer.transform(cv.transform(docs_test))

results=[]
for i in range(tf_idf_vector.shape[0]):
    
    # get vector for a single document
    curr_vector=tf_idf_vector[i]
    
    #sort the tf-idf vector by descending order of scores
    sorted_items=sort_coo(curr_vector.tocoo())

    #extract only the top n; n here is 100
    keywords=extract_topn_from_vector(feature_names,sorted_items,100)
    
    
    results.append(keywords)

df=pd.DataFrame(zip(docs,results),columns=['doc','keywords'])
df
    
    
    
#CLASIFICACION
#NAIVE BAYES


    
    
    



#DECISION TREE





#RANDOM FOREST
