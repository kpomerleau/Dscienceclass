##Project Question and Data Set

**Project Question:**

Is it possible to predict the outcome of a Dota 2 match before that match begins based on hero selection?
		
**Project Data:**

I will be using the Steam API which contains details of Dota 2 matches. Every public Dota 2 match ever played is assigned a match ID. The API allows one to access a significant amount detail about each public match by its ID. Specifically, It contains a list of each of the players (by account ID), which hero each player chose, what team each player was on, the items purchased through the match for each hero, and each player’s performance in regard to kills, deaths, assists, last hits, denies, gold per minute, etc. In addition it has general details about each match: start time, duration, game mode, and league ID (if the match was part of a league).

For the purposes of this project, the most important data is that of each player’s hero selection. In addition, it may be possible to control for certain confounding factors such as game types (some game types allow individuals to “counter-pick” certain heroes).

Link to Match Detail API data: https://wiki.teamfortress.com/wiki/WebAPI/GetMatchDetails

**Interest:**

I am interested in answering this question mainly because I play Dota 2 frequently. It will be interesting to apply data science to a topic I both understand and enjoy. Secondly, Dota 2 is a game played in real time. Many factors go into a "successful" match. It will be interesting to see if there are fundamental parts of the game (beyond skill and luck) that determine the outcome.

		
