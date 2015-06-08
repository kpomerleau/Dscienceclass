1. USing Chipotle.tsv in the data subdirectory:
	navated to the data subdirectory: cd ~/documents/github/dat7/data
	i. look at head and tail. What do the rows and columns mean
		code used:
			'''
			head chipotle
			tail chipotle
			'''
		Rows are each item order. Each item is designed by quantity (how many of each item was ordered), the item's name, the specific choice, and the price
		Columns are variables for each item ordered: order id, quantity, item name, item specifications, and the price

	ii. How many orders do their appear to be?
		code used: 
			'''
			tail chipotle
			'''
		 1834 order
	iii. How many lines are in the file?
		code used:
			'''
			wc chipotle.tsv -l
			'''
		4623 lines
	iv. Which burrito is more popular, steak or chicken?
		code used:
			'''
			grep 'Chicken Burrito' chipotle.tsv | wc -l
			grep 'Steak Burrito' chipotle.tsv | wc -l
			'''
		The chicken burrito was ordered 553 times and the steak burrito was ordered 368 times. 
	v. Do chicken burritos more often have black beans or pinto beans?
		code used:
			'''
			grep 'Pinto Beans' chipotle.tsv | grep 'Chicken Burrito' | wc -l
			grep 'Black Beans' chipotle.tsv | grep 'Chicken Burrito' | wc -l
			'''
		105 of 553 chicken burritos (orders) had pinto beans
		282 of 553 chicken burritos (orders) had black beans
		Chicken burritos more often have black beans
2. Make a list of all CSV and TSV files in the DAT repo
	code used:
		'''
		cd ~/documents/github/dat7
		find -name *.?sv
		'''
	./data/airlines.csv
	./data/chipotle.tsv
	./data/drinks.csv
	./data/imdb_1000.csv
	./data/sms.tsv
	./data/ufo.csv
3. Count the number of occurences of the world 'dictionary' across all files in DAT7 repo
	code used: 
		'''
		grep -r 'dictionary' . -i | wc
		'''
	mentioned 16 times
	
		
