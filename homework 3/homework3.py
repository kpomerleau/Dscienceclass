# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 10:17:39 2015

@author: Kyle
"""

'''
Pandas Homework with IMDB data
'''

import pandas as pd

'''
BASIC LEVEL
'''

# read in 'imdb_1000.csv' and store it in a DataFrame named movies

import csv
movies = pd.read_csv('imdb_1000.csv') 

# check the number of rows and columns

movies.shape

#'979 rows, 6 columns

# check the data type of each column

 movies.dtypes

#'star_rating       float64
#'title              object
#'content_rating     object
#'genre              object
#'duration            int64
#'actors_list        object

# calculate the average movie duration

movies.duration.mean()

#'120.97

# sort the DataFrame by duration to find the shortest and longest movies

movies = movies.sort('duration')
movies.head(1)
movies.tail(1)

#'Shortest: Freaks, 64 minutes
#'Longest: Hamlet, 242 minutes

# create a histogram of duration, choosing an "appropriate" number of bins

movies.duration.plot(kind='hist', bins=8)

# use a box plot to display that same data

movies.duration.plot(kind='box')

'''
INTERMEDIATE LEVEL
'''

# count how many movies have each of the content ratings

movies.content_rating.value_counts()

#'R            460
#'PG-13        189
#'PG           123
#'NOT RATED     65
#'APPROVED      47
#'UNRATED       38
#'G             32
#'PASSED         7
#'NC-17          7
#'X              4
#'GP             3
#'TV-MA          1


# use a visualization to display that same data, including a title and x and y labels

movies.content_rating.value_counts().plot(kind='bar')

# convert the following content ratings to "UNRATED": NOT RATED, APPROVED, PASSED, GP

movies.content_rating.replace(['NOT RATED', 'APPROVED', 'PASSED', 'GP'], 'UNRATED', inplace=True)

# convert the following content ratings to "NC-17": X, TV-MA

movies.content_rating.replace(['X','TV-MA'], 'NC-17', inplace=True)

# count the number of missing values in each column

movies.isnull().sum() 

#'star_rating       0
#'title             0
#'content_rating    3
#'genre             0
#'duration          0
#'actors_list       0

# if there are missing values: examine them, then fill them in with "reasonable" values

movies[movies.content_rating.isnull()]

#'Butch Cassidy and the Sundance Kid is actually PG
#'True Grit is actually G
#'Where Eagles Dare is actually PG

movies.content_rating[movies.title == 'Butch Cassidy and the Sundance Kid'] = 'PG'
movies.content_rating[(movies.title == 'True Grit') & (movies.star_rating == 7.4)] = 'G' 
movies.content_rating[movies.title == 'Where Eagles Dare'] = 'PG'

#This gives a warning, but still works.

# calculate the average star rating for movies 2 hours or longer,

movies.star_rating[movies.duration > 119].mean()

#'7.948

# and compare that with the average star rating for movies shorter than 2 hours

movies.star_rating[movies.duration < 120].mean()

#'7.838

# use a visualization to detect whether there is a relationship between star rating and duration

movies.plot(kind='scatter', x='duration', y='star_rating')

# calculate the average duration for each genre

movies.groupby('genre').duration.mean()

#'Action       126.485294
#'Adventure    134.840000
#'Animation     96.596774
#'Biography    131.844156
#'Comedy       107.602564
#'Crime        122.298387
#'Drama        126.539568
#'Family       107.500000
#'Fantasy      112.000000
#'Film-Noir     97.333333
#'History       66.000000
#'Horror       102.517241
#'Mystery      115.625000
#'Sci-Fi       109.000000
#'Thriller     114.200000
#'Western      136.666667

'''
ADVANCED LEVEL
'''

# visualize the relationship between content rating and duration

movies.groupby('content_rating').duration.mean().plot(kind='bar')

# determine the top rated movie (by star rating) for each genre

movies.groupby('genre').star_rating.max()

movies = movies.sort(['genre','star_rating'])
movies.groupby('genre').tail(1)

# check if there are multiple movies with the same title, and if so, determine if they are actually duplicates

movies.title.duplicated().sum()

#'4 movies share their names

movies[movies.title.duplicated()]

#'True Grit
#'The Girl with the Dragon Tattoo
#'Les Miserables
#'Dracula

#'I know these movies have remakes

movies.duplicated(['title', 'actors_list']).sum()

#'they are not actually duplicates when looking at the actors list

# calculate the average star rating for each genre, but only include genres with at least 10 movies

movies.groupby('genre').count()

#Subset the data. only genres with 10 movies

moviessub = movies[movies.genre.isin(['Action','Adventure','Animation','Biography','Comedy','Crime','Drama','Horror','Mystery'])]

#check to see if it worked

moviessub.groupby('genre').count()

#now find the average star rating!

moviessub.groupby('genre').star_rating.mean()

#Action       7.884559
#Adventure    7.933333
#Animation    7.914516
#Biography    7.862338
#Comedy       7.822436
#Crime        7.916935
#Drama        7.902518
#Horror       7.806897
#Mystery      7.975000

'''
BONUS
'''

# Figure out something "interesting" using the actors data!
