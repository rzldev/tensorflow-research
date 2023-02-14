## TensorFlow Extras ##

import tensorflow as tf
print('\ntensorflow: ', tf.__version__)

## Changing thr datatype of a tensor

# Create a new tensor with default datatype
tensor1 = tf.constant([2.12, 8.67])
print('\ntensor1: {0}\n'.format(tensor1.dtype), tensor1)

tensor2 = tf.constant([[3, 9], [7, 4]])
print('\ntensor2: {0}\n'.format(tensor2.dtype), tensor2)

# Change tensor datatype from float32 to float16
tensor1_float16 = tf.cast(tensor1, dtype=tf.float16)
print('\ntensor1: {0}\n'.format(tensor1_float16.dtype), tensor1_float16)

# Change tensor datatype from int32 to float32
tensor2_float32 = tf.cast(tensor2, dtype=tf.float32)
print('\ntensor2: {0}\n'.format(tensor2_float32.dtype), tensor2_float32)

## One-Hot encoding tensor
indices = [0, 1, 2, 3, 4]

encoded_list = tf.one_hot(indices, depth=len(indices))
print('\nencoded list:\n', encoded_list)

## Tensors and NumPy
import numpy as np
np.random.seed(24)

np_tensor = tf.constant(np.random.randint(1, 8, size=(2, 3)))
print('\nCreated tensor from numpy array: {0}\n'.format(np_tensor.dtype), np_tensor)

np_array = np.array(np_tensor)
print('\nConverted back tensor to numpy array: {0}\n'.format(np_array.dtype), np_array)

## Run tensor operations on GPU
print('\nList of physical device available: ', tf.config.list_physical_devices())
print('List of GPU available: ', tf.config.list_physical_devices('GPU'))

'''
NOTE:
If you have access to CUDA-enabled GPU, TensorFlow will automatically use it
whenever possible.
'''