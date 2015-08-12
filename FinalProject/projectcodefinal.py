# -*- coding: utf-8 -*-
"""
Dota 2 API
"""

"""
This resets the variables
"""
cd ~/Documents/Github
%reset -f

"""
Import pandas and the dota2api package
"""

#from dota2py import api
import dota2api
import pandas as pd
from collections import defaultdict
import requests
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import requests
from time import sleep
import glob
import ast
"""
Initialize the api with my api key
"""

api = dota2api.Initialise("65B6FAD79F62890273994DF5DDC28A2C")

#api.set_api_key("65B6FAD79F62890273994DF5DDC28A2C")

"""
first I need a list of all Dota 2 Champions from the API
"""
#Grab from the API
herosdict = api.get_heroes()

#convert to a dataframe
heroes = pd.DataFrame(herosdict['heroes'])

#choose only hero names and hero id number columns
heroes = heroes.drop('name', axis = 1)

#rename localized_name to name
heroes = heroes.rename(columns={'localized_name':'name'})

heroes.id.value_counts()

#More Testing

#testing for leaver status

def leaver(match):
    for player in match['players']:
        if player['leaver_status'] is not 0:
            return True
        return False

seqnum = 1480382075 #First sequence number: 1480369501 #Latest game downloaded:1480542850
games = []
filenum = 1
while True:
    sleep(5)
    matches = api.get_match_history_by_seq_num(start_at_match_seq_num=seqnum)
       
    vmatches = matches['matches'] #dictionary of matches that can be iterated through
    
    for match in vmatches:
        
        if match['game_mode'] == 22 and match['human_players'] == 10 and match['lobby_type'] == 7 and leaver(match) == False: #game mode two is capitains mode, also only look at matches with 10 players and lobby type 7 isd ranked matches
                                           
              games.append(match)
    
    if len(games) > 500:
        
        #take the games and write them as features
                    
        NUM_HEROES = 112 #This is the max hero id, not the number of heros
        NUM_FEATURES = NUM_HEROES*2
        MATCHES = len(games)

        X = np.zeros((MATCHES, NUM_FEATURES), dtype=np.int8) #1 or zero, whether hero exists
        Y = np.zeros(MATCHES, dtype=np.int8) #Win/Loss
        G = np.zeros((MATCHES, NUM_FEATURES), dtype=int) #GPM, Need to set this to int32! I dunno why, but you gotta
        XP = np.zeros((MATCHES, NUM_FEATURES), dtype=int) #Xp per minute
        
        for i, match in enumerate(games):
    
            Y[i] = 1 if games[i]['radiant_win'] else 0
        
            players = games[i]['players']
        
            for player in players:
                hero_id = player['hero_id'] -1 
                    
                player_slot = player['player_slot']
                if player_slot >= 128: # player slots are 0:4 for Radiant, and 128:132 for Dire.
                    hero_id += NUM_HEROES
            
                X[i, hero_id] = 1
                G[i, hero_id] = player['gold_per_min']
                XP[i, hero_id] = player['xp_per_min']

        #Save the latest 500 matches in three datasets, one for each feature
        
        Xdata = pd.DataFrame(X)
        Gdata = pd.DataFrame(G)
        XPdata = pd.DataFrame(XP) 
        Ydata = pd.DataFrame(Y)
        
        num = str(filenum)
        Xdata.to_csv('Xdata' + num + '.csv')
        Gdata.to_csv('Gdata' + num + '.csv')
        XPdata.to_csv('XPdata' + num + '.csv')
        Ydata.to_csv('Ydata' + num + '.csv')
        
        games = []
        filenum += 1 
    seqnum = vmatches[99]['match_seq_num'] + 1
       
    if filenum == 51:
        
        break
    

#Load the datafiles into memory

#regular Xs

path = r'C:\Users\kep\Documents\Github'    
allFiles = glob.glob(path + '/Xd*.csv')
data = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None)
    list_.append(df)
data = pd.concat(list_)
Xdata = data
Xdata = Xdata.drop('Unnamed: 0', axis =1)

#Ys

path = r'C:\Users\kep\Documents\Github'    
allFiles = glob.glob(path + '/Y*.csv')
data = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None)
    list_.append(df)
data = pd.concat(list_)
Ydata = data
Ydata = Ydata.drop('Unnamed: 0', axis =1)


#descriptive statistics of games

data.head()

#number of wins/loses for radiant, This is the null model

Ydata['0'].mean()
Ydata['0'].sum()

#53.3

#Number of Heroes used in Dataset

times = []
radianttimes = []
for x in range(0,Xdata.shape[1]/2):
    
    r = str(x)    
    d = str(x+112)
    
    times.append(Xdata[r].sum()+Xdata[d].sum())
    radianttimes.append(Xdata[r].sum())
    
times = pd.DataFrame(times)
radianttimes = pd.DataFrame(radianttimes)
times['id'] = times.index + 1
radianttimes['id'] = radianttimes.index+1
test = pd.merge(radianttimes,heroes, on = 'id')
heroesstats = pd.merge(times,heroes, on = 'id')
heroesstats.columns = ['n','id','hero_name']


#Linear Probability Model

X_train, X_test, y_train, y_test = train_test_split(Xdata,Ydata['0'], random_state=1)

linreg = LinearRegression()
linreg.fit(X_train,y_train)

lin_pred_prob = linreg.predict(X_test)
lin_pred_class = np.where(lin_pred_prob >= 0.5, 1, 0)
print metrics.accuracy_score(y_test, lin_pred_class)


fpr, tpr, thresholds = metrics.roc_curve(y_test, lin_pred_prob)
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve, Logistic Regression, Model 1')

print metrics.roc_auc_score(y_test, lin_pred_prob)

#Accuracy = 61.36%
#AUC = 64.76%

#logistic Model

#Just Binary

logreg = LogisticRegression(C=1e9)
logreg.fit(X_train,y_train)

log_pred_class = logreg.predict(X_test)
print metrics.accuracy_score(y_test, log_pred_class)
logscores = cross_val_score(logreg, Xdata, Ydata['0'], cv =40, scoring='accuracy')
print logscores.mean()


log_pred_prob = logreg.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test, log_pred_prob)
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve, Logistic Regression, Model 1')

print metrics.roc_auc_score(y_test, log_pred_prob)

#Decision Trees

#cross val setup

max_depth_range = range(11,20)
Accuracy = []

for depth in max_depth_range:
    tree = DecisionTreeClassifier(max_depth = depth, random_state=1)
    scores = cross_val_score(tree, X, Ydata['0'], cv =10, scoring='accuracy')
    Accuracy.append(np.mean(scores))

plt.plot(max_depth_range, Accuracy)

#A vanilla decision tree only peaks at about 55.2 percent with a depth of 10.

#Random Forests
#looking for N
from sklearn.ensemble import RandomForestClassifier
estimator_range = range(10,500,10)
Accuracy = []

for estimator in estimator_range:
    rf = RandomForestClassifier(n_estimators = estimator, random_state = 1)
    scores = cross_val_score(rf, X, Ydata['0'], cv =10, scoring='accuracy')
    Accuracy.append(np.mean(scores))
plt.plot(estimator_range, Accuracy)

#Looks like N should be around 450 or 500 to maximize accuracy. Accuracy of about 59.9 percent

rf = RandomForestClassifier(n_estimators = 500, random_state = 1)
rfscores = cross_val_score(rf, Xdata, Ydata['0'], cv =40, scoring='accuracy')
rfscores.mean()

rf.fit(X_train, y_train)
rf_pred_prob = rf.predict_proba(X_test)[:,1]
rf_pred_class = rf.predict(X_test)

print metrics.accuracy_score(y_test, rf_pred_class)
print metrics.roc_auc_score(y_test, rf_pred_class)


#Try the model on the International 5 Grand Finals (Radiant Wins)

Xreal = np.zeros((1, 112*2), dtype=np.int8) #Heroes
heroes = [68,17,7,89,72,112+112,112+51,112+25,49+112,112+12]

for hero in heroes:
    Xreal[0,hero-1] = 1

print logreg.predict_proba(Xreal)[:,1]
print rf.predict_proba(Xreal)[:,1]

#63 percent likelihood of victory for Radiant under RandomForest
#58 percent likelihood of victory for Radiant under Logistic Regression

