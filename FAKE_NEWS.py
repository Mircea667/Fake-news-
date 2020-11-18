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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from nltk.corpus import stopwords
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve


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

##LEARNING CURVES

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


fig, axes = plt.subplots(3, 1,figsize=(10, 15))

#X, y = load_digits(return_X_y=True)

title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)

estimator1 = GaussianNB()
plot_learning_curve(estimator1, title, X, y,
                    cv=cv, n_jobs=-1)

title2 = r"Learning Curves (DT)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)

estimator2 = DecissionTreeClassifier()
plot_learning_curve(estimator2, title2, X, y, 
                    cv=cv, n_jobs=-1)

title3 = r"Learning Curves (RF)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)

estimator3 = RandomForestClassifier()
plot_learning_curve(estimator2, title3, X, y,
                    cv=cv, n_jobs=-1)
plt.show()



#lc1=learning_curve(GaussianNB(), X_train2, y_train)
#lc1.show()
#lc2=learning_curve(DecisionTreeClassifier(), X_train2, y_train)
#lc2.show()
#lc3=learning_curve(RandomForestClassifier(), X_train2, y_train )
#lc3.show()
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


