##Basic Analysis of Chipotle Order Dataset

navagated to the data subdirectory

`cd ~/documents/github/dat7/data`
		

**1. Read in the data with csv.reader() and store it in a list of lists called 'data'.**
					
```
import csv
with open('chipotle.tsv', 'rU') as f:
    data = [row for row in csv.reader(f, delimiter = '\t')]
```

**2. Separate the header and data into two different lists.**
				
```
header = data[0]
data = data[1:]
```

**3. Calculate the average price of an order.**
	
```
averageprice = sum([float(row[4][1:]) for row in data])/1834
averageprice = '$'+str(round(averageprice,2))
print averageprice
```

The average price of an order is $18.81

**4. Create a list (or set) of all unique sodas and soft drinks that they sell.**
		
```
soda = []
for row in data:
	if row[2] in ('Canned Soda','Canned Soft Drink'):
                soda.append(row[3]) 
sodaset = set(soda)
print sodaset

```
```
        '[Lemonade]' 
        '[Dr. Pepper]' 
        '[Diet Coke]' 
        '[Nestea]' 
        '[Mountain Dew]' 
        '[Diet Dr. Pepper]'
        '[Coke]'
        '[Coca Cola]' 
        '[Sprite]'
```

**5. Calculate the average number of toppings per burrito.**

code used:

```
toppings = 0
       burritos = 0
       for row in data:
            if 'Burrito' in row[2]:
                toppings += row[3].count(',')+1 'Commas separate the toppings, so I will count those, plus 1
                burritos += 1                   'Need to count the burritos too   
       avgtoppings = float(toppings/burritos)
       print averagetoppings
```

Average of 5 toppings per burrito

**6. Create a dictionary in which the keys represent chip orders and the values represent the total number of orders.
	

```
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
```

		
