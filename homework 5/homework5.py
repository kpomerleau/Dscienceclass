# -*- coding: utf-8 -*-
"""
Created on Sun Jul 05 11:13:04 2015

@author: Kyle
"""

#Homework 5. Only got to bonus question number 8.

%reset -f

cd ~/documents/github/dat7/data

import pandas as pd
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf
import seaborn as sns

#1:  Load the Data

    data = pd.read_csv('yelp.csv')
    data.head()

#2: Explore the Relationships between the three features
    
    sns.pairplot(data, x_vars=['cool','useful','funny'], y_vars='stars', size=6, aspect=0.7, kind='reg')
    
    sns.pairplot(data)

#3: Define Features and Responses

    features = ['cool','useful','funny']
    response = ['stars']
    
    X = data[features]
    y = data[response]

#4: Fit a Linear Regression

    lm = smf.ols(formula='y ~ X', data=data).fit()
    
    lm.params
    lm.conf_int()
    lm.pvalues
    lm.rsquared

    #Do these make sense?

        #Yes. Holding all else constant, 1 cool rating is associated with a 0.27 higher star rating. It is likely that
        #people see the cool rating as a way to react positively to both a review and a business. Both Funny and useful have negative
        #associations (-.147 and -.135). People may find poor reviews funny. It may not be the case that they are laughing at the business
        #instead, those who are more humorous are more likely to write negative reviews. Useful ratings may relate to "oh thanks for warning me."

#5: Train and Test Sets and RMSE

    def train_test_rmse(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        y_pred = linreg.predict(X_test)
        return np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    
train_test_rmse(X, y)

    #Does this make sense? 

        #Yeah- On average, the regression line has an error of approximately 1.18 stars. The model's R squared is very low (.05) an error equal
        #to 20 percent of the possible star ranking range makes sense.

#6: Remove some features, does it improve?

    X1 = data[['cool','useful']]
    X2 = data[['cool']]
    X3 = data[['useful','funny']]
    
    train_test_rmse(X1, y)
    train_test_rmse(X2, y)
    train_test_rmse(X3, y)

    #nope, gets worse.
    
#7 BONUS: Create some new features. I create 6 additional features and test them one at a time as dummy variables. 
    #These features are based on the text contained in the comment. Using dummies i got the RMSE to 1.14.
    #Then I decided to create more variation in my Xs. I did this by creating an index that measured good and bad words in each comment.
    #With this method I was able to get RMSE down to 1.114 and 1.11 and an R-squared of 17% with non-linear features . I think it could improve
    #if i gave more thought to other text patterns...

    #See if sarcastic questions predict star rating (contains question marks)

    data['text'].str.contains('\?').value_counts()
    data['text'].str.count('\?').value_counts()
    
    data['question'] = data['text'].str.contains('\?').count()
    
    data['question'] = pd.get_dummies(data.question, prefix='question').iloc[:, 1:]
    
    
    X4 = data[['cool','useful','funny', 'question']]
    
    sns.pairplot(data, x_vars=['question'], y_vars='stars', size=6, aspect=0.7, kind='reg')
    
    
    lm = smf.ols(formula='y ~ X4', data=data).fit()

    
    lm.params
    lm.conf_int()
    lm.pvalues
    lm.rsquared
    
    #Compare X4 to the original regression

        train_test_rmse(X, y)
        train_test_rmse(X4, y)

    #See if the length of the comment predicts star rating

    data.text.str.len()
    data['comment_len'] = data.text.str.len()
    
    X5 = data[['cool','useful','funny', 'question', 'comment_len']]
    sns.pairplot(data, x_vars=['comment_len'], y_vars='stars', size=6, aspect=0.7, kind='reg')
    
    lm = smf.ols(formula='y ~ X5', data=data).fit()
    
    lm.params
    lm.conf_int()
    lm.pvalues
    lm.rsquared

    #Compare X5 to the original regression

        train_test_rmse(X, y)
        train_test_rmse(X5, y)
        
    #See if "love" predicts star rating
    
    data['love'] = data['text'].str.contains('love')
    data['love'] = pd.get_dummies(data.love, prefix='love').iloc[:, 1:]
    
    
    X6 = data[['cool','useful','funny', 'question', 'comment_len','love']]    
    
    sns.pairplot(data, x_vars=['love'], y_vars='stars', size=6, aspect=0.7, kind='reg')
    
    lm = smf.ols(formula='y ~ X6', data=data).fit()

    lm.params
    lm.conf_int()
    lm.pvalues
    lm.rsquared
    
        train_test_rmse(X6, y)   
    
    #Contains Best
        
    data['best'] = data['text'].str.contains('best')
    data['best'] = pd.get_dummies(data.best, prefix='best').iloc[:, 1:]

    X7 = data[['cool','useful','funny', 'question', 'comment_len','love','best']]
    
    sns.pairplot(data, x_vars=['best'], y_vars='stars', size=6, aspect=0.7, kind='reg')
    
    
    lm = smf.ols(formula='y ~ X7', data=data).fit()
        
    
    lm.params
    lm.conf_int()
    lm.pvalues
    lm.rsquared
    
        train_test_rmse(X7, y)
        train_test_rmse(X6, y)
    
    #See if great is good
    
    data['great'] = data['text'].str.contains('great')
    data['great'] = pd.get_dummies(data.great, prefix='great').iloc[:, 1:]
    
    X8 = data[['cool','useful','funny', 'question', 'comment_len','love','best','great']]    

    sns.pairplot(data, x_vars=['great'], y_vars='stars', size=6, aspect=0.7, kind='reg')
    
    lm = smf.ols(formula='y ~ X8', data=data).fit()
    
    lm.params
    lm.conf_int()
    lm.pvalues
    lm.rsquared
    
        train_test_rmse(X8, y)
        train_test_rmse(X7, y)        
    
    #bad
    
    data['bad'] = data['text'].str.contains('bad')
    data['bad'] = pd.get_dummies(data.bad, prefix='bad').iloc[:, 1:]

    X9 = data[['cool','useful','funny', 'question', 'comment_len','love','best','great','bad']]    

    sns.pairplot(data, x_vars=['bad'], y_vars='stars', size=6, aspect=0.7, kind='reg')

    lm = smf.ols(formula='y ~ X9', data=data).fit()
    
    
    lm.params
    lm.conf_int()
    lm.pvalues
    lm.rsquared
    
        train_test_rmse(X9, y)
        train_test_rmse(X8, y)        

    #Alternative Model (using an index of good/bad words) and allowing for some non linearity

        data['goodindex'] = data['text'].str.count('love|best|great|awesome|amazing|friendly')
        data['badindex'] = data['text'].str.count('bad|terrible|gross|disgusting|\?')

        data['goodindex2'] = data.goodindex**2
        data['badindex2'] = data.badindex**2

        Xindex = data[['cool','useful','funny', 'comment_len','goodindex','badindex']]   
        Xindex2 = data[['cool','useful','funny', 'comment_len','goodindex','goodindex2','badindex','badindex2']]        
        
        sns.pairplot(data, x_vars=['goodindex'], y_vars='stars', size=6, aspect=0.7, kind='reg')
        sns.pairplot(data, x_vars=['badindex'], y_vars='stars', size=6, aspect=0.7, kind='reg')    

        lm = smf.ols(formula='y ~ Xindex', data=data).fit()
        lm = smf.ols(formula='y ~ Xindex2', data=data).fit()


        lm.params
        lm.conf_int()
        lm.pvalues
        lm.rsquared
        
            train_test_rmse(X9, y)
            train_test_rmse(Xindex, y)        
            train_test_rmse(Xindex2, y)

#8: Compare my best (Xindex) to the Null
    
    Xu = []
    
    for x in range(0,10000):
        
        Xu.append(data.stars.mean())
        
    Xu = pd.DataFrame(Xu)
    lm = smf.ols(formula='y ~ Xu', data=data).fit()
    
    train_test_rmse(Xu, y) 
    train_test_rmse(Xindex2, y)
    
    #1.212 vs. 1.11 lol. A lot of work for nothing!