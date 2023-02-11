## Manipulating Data with Pandas ##

import pandas as pd

car_sales = pd.read_csv('../data/car-sales.csv')
car_sales_missing = pd.read_csv('../data/car-sales-missing-data.csv')

## Calling string functions on the data
print('\nCar sales make:\n', car_sales.Make.str.lower())

## Fill in missing data with mean data from that column
car_sales_missing.Doors.fillna(car_sales_missing.Doors.mean(), inplace=True)
print('\ncar_sales_missing.Doors.fillna():\n', car_sales_missing)
car_sales_missing['Odometer'] = car_sales_missing['Odometer'].fillna(
    car_sales_missing['Odometer'].mean())
print('\ncar_sales_missing.Odometer.fillna():\n', car_sales_missing)

## Remove rows contain a missing value
car_sales_missing = pd.read_csv('../data/car-sales-missing-data.csv')
car_sales_dropped = car_sales_missing.dropna()
print('\ncar_sales_dropped:\n', car_sales_dropped)

## Add new column from Series
seats = pd.Series([2, 5, 5, 5, 2])
car_sales['Seats'] = seats
car_sales.Seats.fillna(car_sales.Seats.mode()[0], inplace=True)
print('\nCar seats:\n', car_sales)

## Add new column from Python list
fuel_economy = [8.2, 7.4, 9.0, 4.9, 5.5, 7.6, 4.3, 7.7, 4.7, 9.9]
car_sales['Fuel Per 100KM'] = fuel_economy
print('\nCar fuel /KM:\n', car_sales)

## Add new column with math operation
car_sales['Total Fuel Used'] = car_sales['Odometer (KM)'] / 100 * car_sales['Fuel Per 100KM']
print('\nTotal fuel used:\n', car_sales)

## Add new column from a variable
car_sales['Passed Road Safety'] = True
print('\nCar sales data types:\n', car_sales.dtypes)

## Drop a column
car_sales.drop('Total Fuel Used', axis=1, inplace=True)
print('\nDrop total fuel used:\n', car_sales)

## Shuffle the data
car_sales_shuffled = car_sales.sample(frac=1)
print('\nShuffle data:\n', car_sales_shuffled)
print('\nGet fraction of the data:\n', car_sales.sample(frac=.2))

## Reset the index
car_sales_shuffled.reset_index(drop=True, inplace=True)
print('\nReset shuffled index:\n', car_sales_shuffled)

## Apply function to a certain column
car_sales['Odometer (Mile)'] = car_sales['Odometer (KM)'].apply(lambda x: x * 1.6)
print('\nApply function to a certain column:\n', 
      car_sales.loc[:, ['Make', 'Odometer (KM)', 'Odometer (Mile)']])
