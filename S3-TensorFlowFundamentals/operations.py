## Tensor Operations ##

import numpy as np
import tensorflow as tf
print('\ntensorflow: ', tf.__version__)

## Basic oprations
tensor1 = tf.constant([[10, 7], [4, 8]])

tensor_addition1 = tensor1 + 10
print('\ntensor_addition1: \n', tensor_addition1)
tensor_addition2 = tf.add(tensor1, 20)
print('\ntensor_addition2: \n', tensor_addition2)

tensor_subtraction1 = tensor1 - 2
print('\ntensor_subtraction1: \n', tensor_subtraction1)
tensor_subtraction2 = tf.subtract(tensor1, 4)
print('\ntensor_subtraction2: \n', tensor_subtraction2)

tensor_multiplication1 = tensor1 * 3
print('\ntensor_multiplication1: \n', tensor_multiplication1)
tensor_multiplication2 = tf.multiply(tensor1, 5)
print('\ntensor_multiplication2: \n', tensor_multiplication2)

tensor_division1 = tensor1 / 2
print('\ntensor_division1: \n', tensor_division1)
tensor_division2 = tf.divide(tensor1, 10)
print('\ntensor_division2: \n', tensor_division2)

## Matrix multiplication

# Matrix multiplication with '@' Python operator
tensor2 = tf.constant([[4, 8],
                      [3, 5]])
print('\ntensor2: \n', tensor2)
print('\ntensor1 @ tensor2: \n', (tensor1 @ tensor2))

# Matrix multiplication with tf.matmul()
tensor3 = tf.constant([[4, 8, 3],
                       [3, 5, 9],
                       [7, 4, 1]])
print('\ntensor3: \n', tensor3)
tensor4 = tf.constant(np.random.randint(1, 10, size=(2, 3), dtype=np.int32))
print('\ntensor4: \n', tensor4)
print('\ntf.matmul(tensor3, tensor4): (with reshape)\n', 
      tf.matmul(tensor3, tf.reshape(tensor4, shape=(3, 2))))
print('\ntf.matmul(tensor3, tensor4) (with transpose)): \n', 
      tf.matmul(tensor3, tf.transpose(tensor4)))

'''
NOTE:
Rules that need to be fulfilled if you are going to do matrix multiplication.
1. The inner dimension must match.
2. The resulting matrix has the shape of the outer dimensions.
'''

# Alternative to tf.matmul()
print('\ntf.tensordot(tensor3, tensor4): \n', 
      tf.tensordot(tensor3, tf.transpose(tensor4), axes=1))

'''
NOTE:
*   axes=1 in tensordot() is equivalent to matrix multiplication.
*   Generally when perform matrix multiplication on two tensors and one of the 
    axes doesn't line up, you will transpose (rather than reshape) one of the
    tensors to get satisfy matrix multiplication rules.
'''

## Aggregation
'''
NOTE:
*   Aggregating: condensing them from multiple values down to a smaller amount
    of values.
*   Forms of aggregation:
    1. Minimum
    2. Maximum
    3. Mean
    4. Sum
    5. Etc
'''

# Get absolute value of a tensor
negative_tensor = tf.constant([[-6, -2], [-11, -21]])
print('\nnegative_tensor: \n', negative_tensor)
print('\nabsolute value of negative_tensor: \n', tf.abs(negative_tensor))

# Min, Max, Mean, Sum
print('\nMin value of negative_tensor: ', tf.reduce_min(negative_tensor))
print('\nMax value of tensor3: ', tf.reduce_max(tensor3))
print('\nMean value of tensor4: ', tf.reduce_mean(tensor4))
print('\nSum value of negative_tensor: ', tf.reduce_sum(negative_tensor))
print('\nStandard variance of tensor3: ', 
      tf.math.reduce_std(tf.cast(tensor3, dtype=tf.float32)))

## Get the minimum and maximum data position from a tensor
rank_2_tensor = tf.random.uniform(shape=(2, 4), minval=0, maxval=100, dtype=tf.int32)
print('\nrank_2_tensor: ', rank_2_tensor)

# Minimum data position
min_pos = tf.argmin(rank_2_tensor, axis=1)
print('\nmin_pos: ', min_pos)

# Maximum data position
max_pos = tf.argmax(rank_2_tensor, axis=1)
print('max_pos: ', max_pos)

## Squeezing a tensor (removing all single dimensions)
unsqueezed_tensor = tf.random.uniform(shape=(1, 1, 1, 2, 5), 
                                      minval=0, 
                                      maxval=100, 
                                      dtype=tf.int32)
print('\nunsqueezed_tensor: ', unsqueezed_tensor)

squeezed_tensor = tf.squeeze(unsqueezed_tensor)
print('\nsqueezed_tensor: ', squeezed_tensor)

## More operations (square, square root, log)
print('\nSquare of tensor2: ', tf.square(tensor2))
print('\nSquare root of tensor2: ', tf.sqrt(tf.cast(tensor2, dtype=tf.float32)))
print('\nLog of tensor2: ', tf.math.log(tf.cast(tensor2, dtype=tf.float32)))
