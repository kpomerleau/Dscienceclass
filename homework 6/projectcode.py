# -*- coding: utf-8 -*-
"""
Dota 2 API
"""

"""
This resets the variables
"""

%reset -f

"""
Import pandas and the dota2api package
"""

from dota2py import apiimport pandas as pd

import requests
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import requests
from time import sleep

"""
Initialize the api with my api key
"""
api.set_api_key("XX")

"""
first I need a list of all Dota 2 Champions from the API
"""
#Grab from the API
herosdict = api.get_heroes()['result']

#convert to a dataframe
heroes = pd.DataFrame(herosdict['heroes'])

#choose only hero names and hero id number columns
heroes = heroes.drop('name', axis = 1)

#rename localized_name to name
heroes = heroes.rename(columns={'localized_name':'name'})

heroes.id.max()

#More Testing

#games = api.get_match_history_by_seq_num(start_at_match_seq_num=1428646407)

#Step 1, get a list of most recent 100 matches

starting_match = None
match_ids = []
gamelist = api.get_match_history(start_at_match_id=starting_match,
                                 skill=3,
                                 game_mode=2,
                                 min_players=10,
                                 matches_requested=100
                                 )['result']
                                 
gamelist_id = gamelist['matches']


for i, match in enumerate(gamelist_id):

    match_ids.append(match['match_id'])
    
starting_match = np.array(match_ids).min() - 1


#Step 2, get match details

games = []

for i in match_ids:
    
    gmd = api.get_match_details(i)['result']
    
    games.append(gmd)
    sleep(1.0)
#DATA Cleaning

for i, match in enumerate(games):
    
    players = games[i]['players']

#Create my X and Y variables.
#Got this code from someone else who did a similar project.

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


playerstest = games[0]['players']

for player in playerstest:
    print player['gold_per_min']

for i, match in enumerate(games):
    
    players = games[i]['players']
    
    for player in players:
        hero_id = player['hero_id'] - 1 
        gpm = player['gold_per_min']
        player_slot = player['player_slot']
        if player_slot >= 128: # player slots are 0:4 for Radiant, and 128:132 for Dire.
            hero_id += NUM_HEROES
        
        G[i, hero_id] = gpm
        
        
#Basic Data Analysis
        
        #Dire vs. Radiant Win Rates
        
            