## NumPy Dot Product Vs Element Wise ##

import numpy as np

a1 = np.random.randint(0, 10, size=(5, 3))
a2 = np.random.randint(0, 10, size=(5, 3))

print('\na1: \n', a1)
print('\na2: \n', a2)

## Element Wise multiplication (Hadamard Product)
# With the Hadamard product (element-wise product) you multiply the corresponding 
# components, but do not aggregate by summation, leaving a new vector with the 
# same dimension as the original operand vectors.
print('\na1 * a2: \n', (a1 * a2))

## NumPy Dot Product
# With the dot product, you multiply the corresponding components and add those 
# products together.
a3 = np.dot(a1, a2.T)
print('\na3: \n', a3)


## NumPy Manipulating Array & Dot Product Exercise ##

import numpy as np
import pandas as pd

## Create the sales and prices array
np.random.seed(0)
weekly_sales = np.random.randint(0, 20, size=(5, 3))
weekly_sales_df = pd.DataFrame(weekly_sales,
                               index=['Mon', 'Tues', 'Wed', 'Thurs', 'Fri'],
                               columns=['Almond Butter', 'Peanut Butter', 'Cashew Butter'])
print('\nweekly_sales_df: \n', weekly_sales_df)

# product_prices = np.random.randint(10, 100, size=(1, 3))
product_prices = np.array([10, 8, 12])
product_prices_df = pd.DataFrame(product_prices.reshape(1, 3),
                                 index=['Price'],
                                 columns=['Almond Butter', 'Peanut Butter', 'Cashew Butter'])
print('\nproduct_prices_df: \n', product_prices_df)

## Create total sales
total_sales = weekly_sales.dot(product_prices.T)
print('\ntotal_sales: \n', total_sales)

## Complete data
weekly_sales_df['Total Sales ($)'] = total_sales
print('\nweekly_sales_df: \n', weekly_sales_df)
print('\nproduct_prices_df: \n', product_prices_df)
