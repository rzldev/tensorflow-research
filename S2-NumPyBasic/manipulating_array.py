## Manipulating NumPy Arrays ##

import numpy as np
import time

zeros = np.zeros((3))
ones = np.ones((3))
a1 = np.array([3, 8, 11])
a2 = np.random.randint(0, 10, size=(2, 3))
a3 = np.random.randint(0, 10, size=(2, 3, 3))

## Manipulating data array with math operation
print('\na1 + ones: \n', (a1 + ones))
print('\na2 - ones: \n', (a2 - ones))
print('\na2 * a1: \n', (a2 * a1))
print('\na2 ** 2: \n', (a2 ** 2))

## Using NumPy functions to manipulate the data
print('\nnp.square(a2): \n', np.square(a2))
print('\nnp.add(a1, a2): \n', np.add(a1, a2))
print('\nnp.exp(a1): \n', np.exp(a1))
print('\nnp.log(a2): \n', np.log(a2))

## Aggregation
# Aggregation = performing the same operation on a number of things
print('\nsum(a1): \n', sum(a1))
print('\nnp.sum(a1): \n', np.sum(a1))

massive_array = np.random.rand(10000000)
start = time.perf_counter()
sum(massive_array)
end = time.perf_counter()
print('\ntime sum(massive_array): {0:.3f} ms'.format((end - start) * 1000))

start = time.perf_counter()
np.sum(massive_array)
end = time.perf_counter()
print('\ntime np.sum(massive_array): {0:.3f} ms'.format((end - start) * 1000))

print('\nnp.mean(a2): ', np.mean(a2))
print('\nnp.max(a2): ', np.max(a2))
print('\nnp.min(a2): ', np.min(a2))

# Standard Deviation: a measure of how spread out a group of numbers is from the mean
print('\nnp.std(a2): ', np.std(a2))

# Variance: Measure the average degree to which each number is different to mean
# Higher Variance: Wider range of numbers
# Lower Variance: Lower range of numbers
print('\nnp.var(a2): ', np.var(a2))

# Standard Deviation: a squareroot of variance
print('\nnp.sqrt(np.var(a2)): ', np.sqrt(np.var(a2)))

## Standard Deviation & Variance
high_var_array = np.array([100, 400, 300, 600, 200, 800, 500, 1000])
low_var_array = np.array([1, 3, 3, 6, 2, 8, 5, 10])

print('\nhigh_var_array: ', high_var_array)
print('standard deviation: {0} \nvariance: {1} \nmean: {2}'
      .format(high_var_array.std(), high_var_array.var(), high_var_array.mean()))

print('\nlow_var_array: ', low_var_array)
print('standard deviation: {0} \nvariance: {1} \nmean: {2}'
      .format(low_var_array.std(), low_var_array.var(), low_var_array.mean()))

import matplotlib.pyplot as plt
plt.hist(high_var_array)
plt.show()

plt.hist(low_var_array)
plt.show()

## Reshape & Transpose
# Reshape
a2_reshaped = a2.reshape(2, 3, 1)
print('\na2_reshaped: \n', a2_reshaped)
print('\na2_reshaped * a3: \n', (a2_reshaped * a3))

# Transpose
a2_transposed = a2.T
a3_transposed = a3.T
print('\na2 shape: {0}, a2_transposed shape: {1}\n'
      .format(a2.shape, a2_transposed.shape), a2_transposed)
print('\na3 shape: {0}, a3_transposed shape: {1}\n'
      .format(a3.shape, a3_transposed.shape), a3_transposed)
