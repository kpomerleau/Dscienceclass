'''
Python Homework with Chipotle data
https://github.com/TheUpshot/chipotle
'''
'working directory:
'cd ~/documents/github/DAT7/data

'''
BASIC LEVEL
PART 1: Read in the data with csv.reader() and store it in a list of lists called 'data'.
Hint: This is a TSV file, and csv.reader() needs to be told how to handle it.
      https://docs.python.org/2/library/csv.html
'''
import csv
with open('chipotle.tsv', 'rU') as f:
    data = [row for row in csv.reader(f, delimiter = '\t')]


'''
BASIC LEVEL
PART 2: Separate the header and data into two different lists.
'''
header = data[0]
data = data[1:]

'''
INTERMEDIATE LEVEL
PART 3: Calculate the average price of an order.
Hint: Examine the data to see if the 'quantity' column is relevant to this calculation.
Hint: Think carefully about the simplest way to do this!
'''
    '1. Remove dollar signs, 2. add all item prices together, 3. divide by number of orders
    ' in the dataset (1834)    
    
        averageprice = sum([float(row[4][1:]) for row in data])/1834
        averageprice = '$'+str(round(averageprice,2)) ' This isnt totally necessary, but I wanted to add the dollar sign back in
        print averageprice

        '$18.81

'''
INTERMEDIATE LEVEL
PART 4: Create a list (or set) of all unique sodas and soft drinks that they sell.
Note: Just look for 'Canned Soda' and 'Canned Soft Drink', and ignore other drinks like 'Izze'.
'''
        soda = []
        for row in data:
            if row[2] in ('Canned Soda','Canned Soft Drink'):
                soda.append(row[3]) 
        sodaset = set(soda)
        print sodaset

        '[Lemonade]' 
        '[Dr. Pepper]' 
        '[Diet Coke]' 
        '[Nestea]' 
        '[Mountain Dew]' 
        '[Diet Dr. Pepper]'
        '[Coke]'
        '[Coca Cola]' 
        '[Sprite]'

'''
ADVANCED LEVEL
PART 5: Calculate the average number of toppings per burrito.
Note: Let's ignore the 'quantity' column to simplify this task.
Hint: Think carefully about the easiest way to count the number of toppings!
'''

        toppings = 0
        burritos = 0
        for row in data:
             if 'Burrito' in row[2]:
                 toppings += row[3].count(',')+1 'Commas separate the toppings, so I will count those, plus 1
                 burritos += 1                   'Need to count the burritos too   
        avgtoppings = float(toppings/burritos)
        print averagetoppings

        'Average of 5 toppings per burrito

'''
ADVANCED LEVEL
PART 6: Create a dictionary in which the keys represent chip orders and
  the values represent the total number of orders.
Expected output: {'Chips and Roasted Chili-Corn Salsa': 18, ... }
Note: Please take the 'quantity' column into account!
Optional: Learn how to use 'defaultdict' to simplify your code.
'''

    import collections

    chips = []
    for row in data:
        if 'Chips' in row [2]:
            chips.append(row[2])
    
    chips = [row.replace('-',' ') for row in chips] 'Some orders are miscoded. Fixed that with a simple replace
    
    corders = collections.defaultdict(int)
    for row in chips:
        corders[row] += 1
        
    print corders        

'''
BONUS: Think of a question about this data that interests you, and then answer it!
'''