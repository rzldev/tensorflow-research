## NumPy Arrays ##

import numpy as np

## Creating a NumPy array
sample_array = np.array([1, 2, 3])
print('\nsample_array: {0}\n'.format(type(sample_array)), sample_array)

ones = np.ones((2, 3))
print('\nones: {0}\n'.format(type(ones)), ones)

zeros = np.zeros((3, 3))
print('\nzeros: {0}\n'.format(type(zeros)), zeros)

range_array = np.arange(0, 20, 3)
print('\nrange_array: {0}\n'.format(type(range_array)), range_array)

random_array1 = np.random.randint(0, 50, size=(3, 3))
print('\nrandom_array1: {0}\n'.format(type(random_array1)), random_array1)

random_array2 = np.random.random((3, 3))
print('\nrandom_array2: {0}\n'.format(type(random_array2)), random_array2)

random_array3 = np.random.rand(3, 3)
print('\nrandom_array3: {0}\n'.format(type(random_array3)), random_array3)

## Set NumPy random seed
random_array4 = np.random.rand(3, 3)
print('\nrandom_array4: {0}\n{1}\n*the value of random_array4 will always change'
      .format(type(random_array4), random_array4))

np.random.seed(123)
random_array5 = np.random.rand(3, 3)
print('\nrandom_array5: {0}\n{1}\n*the value of random_array5 won\'t change no matter how many you rerun this program\n'
      .format(type(random_array5), random_array5))

## Unique NumPy array
new_array = np.random.randint(0, 3, size=(1, 6))
print('\nnew_array: \n', new_array)
unique_array = np.unique(new_array)
print('\nunique_array from new_array: {0}\n'.format(type(unique_array)), 
      unique_array)

## Get data from array
random_array6 = np.random.randint(0, 10, size=(3, 3, 3))
print('\nrandom_array6: \n', random_array6)

# From position
print('\nposition random_array6[1]: \n', random_array6[1])
print('\nposition random_array6[2, 0]: \n', random_array6[2, 0])
print('\nposition random_array6[0, 1, 2]: \n', random_array6[0, 1, 2])

# Slicing
print('\nslicing random_array6[1:]: \n', random_array6[1:])
print('\nslicing random_array6[2:, :1]: \n', random_array6[2:, :1])
print('\nslicing random_array6[:, :, :2]: \n', random_array6[:, :, :2])

# Get the first inner n-data
print('\nfirst 2 data of random_array6: \n', random_array6[:, :, :2])
