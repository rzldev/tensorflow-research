## TensorFlow Fundamentals ##

import tensorflow as tf
print('\ntensorflow: ', tf.__version__)

## Create a tensor with tf.Variable()
changeable_tensor = tf.Variable([10, 20])
unchangeable_tensor = tf.constant([20, 10])
print('\nchangeable_tensor: ', changeable_tensor)
print('unchangeable_tensor: ', unchangeable_tensor)

try:
    changeable_tensor[0].assign(30)
    print('\nsuccess change changeable_tensor data: \n', changeable_tensor)
except:
    print('\nfailed to change changeable_tensor')

try:
    unchangeable_tensor[1].assign(30)
    print('\nsuccess change unchangeable_tensor data: \n', unchangeable_tensor)
except:
    print('\nfailed to change unchangeable_tensor')
    

## Create random tensors
random1 = tf.random.Generator.from_seed(23)
random1 = random1.normal(shape=(3, 2))
random2 = tf.random.Generator.from_seed(23)
random2 = random2.normal(shape=(3, 2))

print('\nrandom1: \n', random1)
print('\nrandom2: \n', random2)
print('\nrandom1 == random2: \n', (random1 == random2))

## Shuffle the order of elements in tensor
unshuffled_tensor = tf.constant([[5, 2, 8],
                      [4, 9, 1],
                      [5, 2, 3]])
print('\nunshuffled_tensor: \n', unshuffled_tensor)

shuffled_tensor1 = tf.random.shuffle(unshuffled_tensor)
print('\nshuffled_tensor1: \n', shuffled_tensor1)

# Set random seed so the shuffled value will be the same
tf.random.set_seed(12) # global-level random seed
shuffled_tensor2 = tf.random.shuffle(unshuffled_tensor)
print('\nshuffled_tensor2: \n', shuffled_tensor2)

shuffled_tensor3 = tf.random.shuffle(unshuffled_tensor, seed=12) # operation-level random seed
print('\nshuffled_tensor3: \n', shuffled_tensor3)

'''
NOTE:
Shuffle a tensor is valueable when you want to shuffle your data so the inherent order 
                  doesn't effect learning
'''

## Create tensor with NumPy array
import numpy as np

# Create a tensor of all ones
tensor_ones = tf.ones([4, 6])
print('\ntensor_ones: \n', tensor_ones)

# Create a tensor of all zeros
tensor_zeros = tf.zeros(shape=(3, 2, 4))
print('\ntensor_zeros: \n', tensor_zeros)

vector = np.arange(1, 25, dtype=np.int32)
tensor_from_numpy = tf.constant(vector, shape=(3, 2, 4))
print('\ntensor_from_numpy: \n', tensor_from_numpy)

'''
NOTE:
The main differencee between NumPy arrays and TensorFlow tensors is that tensors
can be run on a GPU (much faster for numerical computing).
'''

## Tensor attributes
'''
NOTE:
when dealing with tensors you probably want to be aware of this following attributes:
1. Shape
2. Rank
3. Axis or Dimension
4. Size
'''

rank_4_tensor = tf.zeros(shape=[2, 3, 4, 5])
print('\nrank_4_tensor: \n', rank_4_tensor)

print('\nDatatype of every element: ', rank_4_tensor.dtype)
print('Number of dimensions (rank): ', rank_4_tensor.ndim)
print('Shape of tensor: ', rank_4_tensor.shape)
print('Elements along the 0 axis: ', rank_4_tensor.shape[0])
print('Elements along the last axis: ', rank_4_tensor.shape[-1])
print('Total number of elements in our tensor: ', tf.size(rank_4_tensor))
print('Total number of elements in our tensor: ', tf.size(rank_4_tensor).numpy())

## Indexing tensor
rank_3_tensor = tf.constant(np.arange(1, 101, dtype=np.int32), shape=(5, 4, 5))
print('\nrank_3_tensor: \n', rank_3_tensor)
print('\nThe first 2 elements of each dimension: \n', rank_3_tensor[:2, :2, :2])
print('\nGet the first element of each dimension from each index except for the final one: \n',
      rank_3_tensor[:1, :1])
print('\nGet the first element of each dimension from each index except for the second one: \n',
      rank_3_tensor[:1, :, :1])

## Expanding tensor
rank_2_tensor = tf.constant(np.random.randint(1, 10, size=(3, 2)))
print('\nrank_2_tensor: \n', rank_2_tensor)

# Add in extra dimension to our rank 2 tensor
not_rank_2_anymore = rank_2_tensor[..., tf.newaxis]
print('\nNew rank 3 tensor: \n', not_rank_2_anymore)

# Alternative to tf.newaxis
print('\nAlternative to tf.newaxis: \n', tf.expand_dims(rank_2_tensor, axis=-1))
print('\nAlternative to tf.newaxis: \n', tf.expand_dims(rank_2_tensor, axis=0))





