# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:35:25 2020

@author: PORTATIL
"""
import os
import glob
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from nltk.corpus import stopwords



# Cargar CSV

df = pd.read_csv(r"C:\Users\PORTATIL\Desktop\TFG\dataset_tfg\isot_dataset\combinado.csv")

#Convert NaN values to empty string
nan_value = float("NaN")

df.replace("", nan_value, inplace=True)

df.dropna( inplace=True)

print((df.columns))


# Definición de los datos

X = df['title'].astype(str) + ' ' + df['body'].astype(str)
y = df['Category']
#y_list=y.tolist()

# Dividir los datos 70% train 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70,test_size=0.30)

#Aplicar TFIDF.
tfIdfVectorizer=TfidfVectorizer(smooth_idf=True,max_features=1000,use_idf=True)
#tfIdfVectorizer = TfidfVectorizer()



X_vectorizer = tfIdfVectorizer.fit_transform(X_train)
X_vector=tfIdfVectorizer.transform(X_test)

X_train2=X_vectorizer.toarray()
X_test2=X_vector.toarray()


#ENTRENAMIENTO NAIVE BAYES
# Definir modelo de clasificación, Naive Bayes, Decision Tree y Random Forest. 
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier=naive_bayes_classifier.fit(X_train2, y_train)

pred_NB = naive_bayes_classifier.predict(X_test2)

#ENTRENAMIENTO DECISION TREE.
decision_tree_classifier=DecisionTreeClassifier()
decision_tree_classifier=decision_tree_classifier.fit(X_train2, y_train)

pred_DT=decision_tree_classifier.predict(X_test2)


#tree.plot_tree(decision_tree) 

#ENTRENAMIENTO RANDOM FOREST
random_forest=RandomForestClassifier(n_estimators=3)
random_forest=random_forest.fit(X_train2,y_train)

pred_RF=random_forest.predict(X_test2)

# compute the performance measures
score1 = metrics.accuracy_score(y_test,pred_NB )
score2 = metrics.accuracy_score(y_test,pred_DT)
score3 = metrics.accuracy_score(y_test,pred_RF)
#MATRIZ DE CONFUSION DEL NAIVE BAYES
print("accuracy:   %0.3f" % score1)

print(metrics.classification_report(y_test, pred_NB,
                                            target_names=['Positive', 'Negative']))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test, pred_NB))

print('------------------------------')
#MATRIZ DE CONFUSION DECISION TREE
print("accuracy:   %0.3f" % score2)

print(metrics.classification_report(y_test,pred_DT ,
                                            target_names=['Positive', 'Negative']))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test, pred_DT))

print('------------------------------')

#MATRIZ DE CONFUSION RANDOM FOREST
print("accuracy:   %0.3f" % score3)

print(metrics.classification_report(y_test, pred_RF,
                                            target_names=['Positive', 'Negative']))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test,pred_RF))

print('------------------------------')


# Entrenar los modelos con X_train e y_train

# Evaluar los modelos usando X_test e y_test