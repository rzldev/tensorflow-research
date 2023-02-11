## Viewing and Selecting Data with Pandas ##

import pandas as pd
import matplotlib.pyplot as plt

car_sales = pd.read_csv('../data/car-sales.csv');
animals = pd.Series(['Cat', 'Dog', 'Snake', 'Horse', 'Elephant', 'Tiger', 'Giraffe'],
                    index=[6, 1, 2, 1, 7, 4, 6])
car_prices = car_sales.Price.str.replace(r'[$,]', '', regex=True)
car_prices = car_prices.replace('\.\d*', '', regex=True).astype(int)

## Getting top rows data
print('\nGetting top rows:')
print(animals.head())
print(car_sales.head(2))

## Getting bottom rows data
print('\nGetting bottom rows:')
print(car_sales.tail())
print(animals.tail(4))

## Getting data from it's index
print('\nGetting data from it\'s index:')
print(animals.loc[6])

## Getting data from it's position
print('\nGetting data from it\'s position:')
print(animals.iloc[6])

## Slicing with .loc and .iloc
print('\nSlicing with .loc:')
print(car_sales.loc[:3])
print('\nSlicing with .iloc:')
print(animals.iloc[3:])

## Gettin data based on the column
print('\nGetting data from a single column:')
print(car_sales['Odometer (KM)'])
print(car_sales.Price)

## Gettin data based on the column with filter
print('\nGetting data from a single column with filter:')
print(car_sales[car_sales['Make'] == 'Toyota'])
print(car_prices[car_prices < 5000])

## Compare data from 2 columns
print('\nCompare data from 2 columns')
print(pd.crosstab(car_sales.Make, car_sales['Colour']))

## Compare more than 2 columns
print('\nCompare data from more than 2 columns')
print(car_sales.groupby(['Make']).mean(numeric_only=True))

## Showing data plot
plt.plot(car_sales['Odometer (KM)'])
plt.ylabel('Odometer (KM)')
plt.show()

plt.hist(car_sales['Odometer (KM)'])
plt.ylabel('Odometer (KM)')
plt.show()

plt.plot(car_prices)
plt.ylabel('Prices')
plt.show()
