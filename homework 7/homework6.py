# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 12:36:41 2015

@author: Kyle
"""

%reset -f

cd ~/documents/github/dat7/data

import pandas as pd
import csv
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
#1:  Load the Data

data = pd.read_csv('yelp.csv')
data.head()

#2 Only include 1 and 5 star reviews

yelp = data[(data.stars == 5) | (data.stars == 1)]

yelp.head()
yelp.stars.value_counts()


#3 Train test set

X_train, X_test, y_train, y_test = train_test_split(yelp.text, yelp.stars, random_state=1)
X_train.shape
X_test.shape

#make my y test and y train 1 = 5 and 0 = 1

y_test[y_test == 1] = 0
y_test[y_test == 5] = 1

y_train[y_train == 1] = 0
y_train[y_train == 5] = 1


#4 Count vectorize the X variable

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
#vect.fit(X_train)
#vect.fit(X_test)

X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

X_test_dtm = X_test_dtm.toarray()
X_train_dtm = X_train_dtm.toarray()

#5 Run Model, Calculate the accuracy of the model

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)

#accuracy of the model
    
print metrics.accuracy_score(y_test, y_pred_class)
    
#6 Calculate the Area under the curve:

y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_pred_prob
print metrics.roc_auc_score(y_test, y_pred_prob)

#7 plot the ROC curve

falsepr, truepr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(falsepr, truepr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')


#8 Print the confusion matrix

print metrics.confusion_matrix(y_test, y_pred_class)

#9 look at some of the false positives and false negatives

#false positives

X_test[y_test < y_pred_class]

#False Negatives

X_test[y_test > y_pred_class]

#both false positives and negatives are generally LONG. The model may have trouble when you throw a wall
#of text at it. False positives also look kind of sarcastic, or they are telling you how much better an
#alternative restaurant is. This could easily be confused with a positive review.

#10 What balances Sensitivity and Specificity 

#we are going to calculate the linear distance from 0,1 to each point on the ROC curve.

distance = np.array(((((1-truepr)**2)+((falsepr)**2))**1/2))

test = pd.DataFrame(np.column_stack((thresholds,distance)), columns = ['thresholds','distance'])

test.thresholds[test.distance == test.distance.min()]

#Balanced Threshold is .996






