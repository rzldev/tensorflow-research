## TensorFlow Introduction ##

## Import TensorFlow library
import tensorflow as tf
print('\ntensorflow: ', tf.__version__)

## Create tensors with tf.constant()
scalar = tf.constant(7)
print('\nscalar:', scalar)

## Check a number of dimensions of a tensor
print('dimension: ', scalar.ndim)

## Create a vector (1-dimensional array)
vector = tf.constant([10, 20])
print('\nvector: ', vector)
print('dimension: ', vector.ndim)

## Create a matrix (more than 1-dimensional array)
matrix = tf.constant([[7, 11],
                      [3, 16]])
print('\nmatrix: ', matrix)
print('dimension: ', matrix.ndim)
print('datatype: ', matrix.dtype)

## Create a matrix with specific datatype
matrix2 = tf.constant([[8, 19],
                      [5, 9]], dtype=tf.float32)
print('\nmatrix2: ', matrix2)
print('dimension: ', matrix2.ndim)
print('datatype: ', matrix2.dtype)

## Create a tensor
tensor = tf.constant([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]],
                      [[10, 11, 12],
                       [13, 14, 15],
                       [16, 17, 18]]])
print('\ntensor: ', tensor)
print('dimension: ', tensor.ndim)

'''
NOTE:
* scalar: a single number
* vecotr: a number with direction (e.g. wind speed and direction)
* matrix: 2-dimensional or more array
* tensor: n-dimensional array (when n can be any number, a 0-dimensional tensor
                               is a scalar, and 1-dimensional tensor is a vector)
'''
