## NumPy Introduction ##

import numpy as np
import pandas as pd

## NumPy's main datatype is ndarray
## Creating NumPy array
# 1-dimensional NumPy array (Array, Vector)
array1 = np.array([3, 9, 2])
print('\n1-dimensional NumPy array:\n', array1)

# 2-dimensional or more NumPy array (Array, Matrix)
array2 = np.array([[2, 4, 7],
                   [7, 5, 1]])
print('\n2-dimensional NumPy array:\n', array2)
array3 = np.array([[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]],
                   [[10, 11, 12],
                    [13, 14, 15],
                    [16, 17, 18]]])
print('\n3-dimensional NumPy array:\n', array3)

## NumPy's attributes
print('''\narray1 shape: {0} \narray2 shape: {1} \narray3 shape: {2}'''
      .format(array1.shape, array2.shape, array3.shape))
print('''\narray1 dimension: {0} \narray2 dimension: {1} \narray3 dimension: {2}'''
      .format(array1.ndim, array2.ndim, array3.ndim))
print('''\narray1 datatype: {0} \narray2 datatype: {1} \narray3 datatype: {2}'''
      .format(array1.dtype, array2.dtype, array3.dtype))
print('''\narray1 size: {0} \narray2 size: {1} \narray3 size: {2}'''
      .format(array1.size, array2.size, array3.size))
print('''\narray1 type: {0} \narray2 type: {1} \narray3 type: {2}'''
      .format(type(array1), type(array2), type(array3)))

## Create a DataFrame from NumPy array
df = pd.DataFrame(array2)
print('\nDataFrame:', type(df))
print(df)
