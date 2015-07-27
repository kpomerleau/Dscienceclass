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

seqnum = 1480494250 #First sequence number: 1480369501 #Latest game downloaded:1480542850
games = []
filenum = 36
while True:
    sleep(5)
    matches = api.get_match_history_by_seq_num(start_at_match_seq_num=seqnum)
       
    vmatches = matches['matches']
    
    for match in vmatches:
        
        if match['game_mode'] == 22 and match['human_players'] == 10:
            
            games.append(match)
    
    if len(games) > 500:
        
        #take the games and write them as features
                    
        NUM_HEROES = 112 #This is the max hero id, not the number of heros
        NUM_FEATURES = NUM_HEROES*2
        MATCHES = len(games)

        X = np.zeros((MATCHES, NUM_FEATURES), dtype=np.int8) #Heroes
        Y = np.zeros(MATCHES, dtype=np.int8) #Win/Loss
        G = np.zeros((MATCHES, NUM_FEATURES), dtype=int) #GPM, Need to set this to int32! I dunno why, but you gotta
        XP = np.zeros((MATCHES, NUM_FEATURES), dtype=int)
        
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
    

#Load all the datafiles into memory

#Gold Xs

path = r'C:\Users\kep\Documents\Github'    
allFiles = glob.glob(path + '/G*.csv')
data = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None)
    list_.append(df)
data = pd.concat(list_)
Gdata = data
Gdata = Gdata.drop('Unnamed: 0', axis =1)

#XP Xs

path = r'C:\Users\kep\Documents\Github'    
allFiles = glob.glob(path + '/XP*.csv')
data = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None)
    list_.append(df)
data = pd.concat(list_)
XPdata = data
XPdata = XPdata.drop('Unnamed: 0', axis =1)

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

#number of wins/loses for radient, This is the null model

Ydata['0'].mean()
Ydata['0'].sum()

#52.3

#Number of Heroes used in Dataset

times = []
radianttimes = []
herogold = []
for x in range(0,Xdata.shape[1]/2):
    
    r = str(x)    
    d = str(x+112)
    
    times.append(Xdata[r].sum()+Xdata[d].sum())
    radianttimes.append(Xdata[r].sum())
    herogold.append(Gdata[r].mean()+Gdata[d].mean())
    
times = pd.DataFrame(times)
radianttimes = pd.DataFrame(radianttimes)
herogold = pd.DataFrame(herogold)
herogold['id'] = herogold.index + 1
times['id'] = times.index + 1
radianttimes['id'] = radianttimes.index+1
test = pd.merge(radianttimes,heroes, on = 'id')
heroesstats = pd.merge(times,heroes, on = 'id')
heroesstats.columns = ['n','id','hero_name']
heroesgold = pd.merge(herogold,heroes, on = 'id')
heroesgold.columns = ['n','id','hero_name']

#Linear Probability Model

X_train, X_test, y_train, y_test = train_test_split(Xdata,Ydata, random_state=1)

linreg = LinearRegression()
linreg.fit(X_train,y_train)

y_pred = linreg.predict(X_test)
y_pred_class = np.where(y_pred >= 0.5, 1, 0)
print metrics.accuracy_score(y_test, y_pred_class)

#logistic Model

#Just Binary

logreg = LogisticRegression(C=1e9)
logreg.fit(X_train,y_train)

y_pred_class = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)

y_pred_prob = logreg.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve, Logistic Regression, Model 1')

print metrics.roc_auc_score(y_test, y_pred_prob)

#Gold Controls

X_train, X_test, y_train, y_test = train_test_split(Gdata,Ydata, random_state=1)

logreg = LogisticRegression(C=1e9)
logreg.fit(X_train,y_train)

y_pred_class = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)

y_pred_prob = logreg.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve, Logistic Regression, Model 2')

print metrics.roc_auc_score(y_test, y_pred_prob)

#XP Controls

X_train, X_test, y_train, y_test = train_test_split(XPdata,Ydata, random_state=1)

logreg = LogisticRegression(C=1e9)
logreg.fit(X_train,y_train)

y_pred_class = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)

y_pred_prob = logreg.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


print metrics.roc_auc_score(y_test, y_pred_prob)


#Step 1, get a list of most recent 100 matches

#games = []

#for i in data.match_ids:
#    last_match = i
#    gmd = api.get_match_details(i)
    
#    games.append(gmd)
#    sleep(1.0)


#starting_match = None
#x = 0
#match_ids = []

#while True:
    
#    y = 0
        
#    while True: 
        
#        gamelist = api.get_match_history(start_at_match_id=starting_match,
#                                         skill=3,
#                                         game_mode=2,
#                                         min_players=10
#                                         )['result']
                                         
#        gamelist_id = gamelist['matches']
        
        
#        for i, match in enumerate(gamelist_id):
        
#            match_ids.append(match['match_id'])
            
#        starting_match = np.array(match_ids).min() - 1
    
#        y += 100
        
#        if y == 500:
            
#            break
    
#    sleep(1200.0)    
    
#    x += 1    
    
#    if x == 1500:
        
#        break

#Step 2, get match details

#games = []

#for i in match_ids:
    
#    gmd = api.get_match_details(i)['result']
    
#    games.append(gmd)
#    sleep(1.0)
#DATA Cleaning

#for i, match in enumerate(games):
    
#    players = games[i]['players']

#Create my X and Y variables.
#Got this code from someone else who did a similar project.

#NUM_HEROES = 112 #This is the max hero id, not the number of heros
#NUM_FEATURES = NUM_HEROES*2
#MATCHES = len(games)

#X = np.zeros((MATCHES, NUM_FEATURES), dtype=np.int8) #Heroes
#Y = np.zeros(MATCHES, dtype=np.int8) #Win/Loss
#G = np.zeros((MATCHES, NUM_FEATURES), dtype=int) #GPM, Need to set this to int32! I dunno why, but you gotta
#XP = np.zeros((MATCHES, NUM_FEATURES), dtype=int)

#for i, match in enumerate(games):
    
#    Y[i] = 1 if games[i]['radiant_win'] else 0
    
#    players = games[i]['players']
    
#    for player in players:
#        hero_id = player['hero_id'] -1 
            
#        player_slot = player['player_slot']
#        if player_slot >= 128: # player slots are 0:4 for Radiant, and 128:132 for Dire.
#            hero_id += NUM_HEROES
    
#        X[i, hero_id] = 1
#        G[i, hero_id] = player['gold_per_min']
#        XP[i, hero_id] = player['xp_per_min']


#playerstest = games[0]['players']

#for player in playerstest:
#    print player['gold_per_min']

#for i, match in enumerate(games):
    
#    players = games[i]['players']
    
#    for player in players:
#        hero_id = player['hero_id'] - 1 
#        gpm = player['gold_per_min']
#        player_slot = player['player_slot']
#        if player_slot >= 128: # player slots are 0:4 for Radiant, and 128:132 for Dire.
#            hero_id += NUM_HEROES
        
#        G[i, hero_id] = gpm
        
        
#Basic Data Analysis
        
        #Dire vs. Radiant Win Rates
        
            