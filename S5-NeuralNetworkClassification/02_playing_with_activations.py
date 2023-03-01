## Playing With Existing Activations ##
'''
NOTE:
This section only contains experiments on activation functions. Some functions
will be inside the my_functions.py.
'''

import tensorflow as tf
import my_functions as myfunc
from tensorflow.keras.activations import linear, relu, softmax, sigmoid

print('\ntensorflow: ', tf.__version__)
tf.random.set_seed(42)

## Creating samlple data
sample_data = tf.range(-10, 10, dtype=tf.float32)
print('\nSample data: ', sample_data)

## Linear activation
linear_data = myfunc.linear(sample_data)
print('\nLinear data: ', linear_data[:10])
print('\nMatched Linear activation data: {0}'.format(tf.reduce_all(tf.equal(linear(sample_data),
                                                                            linear_data))))
myfunc.plot_sample_data(linear_data, title='Linear Data', c='b')

## ReLu activation
relu_data = myfunc.relu(sample_data)
print('\nReLu data: ', relu_data[:10])
print('\nMatched ReLu activation data: {0}'.format(tf.reduce_all(tf.equal(relu(sample_data), 
                                                                          relu_data))))
myfunc.plot_sample_data(relu_data, title='ReLu Data', c='r')

## Softmax Activation
softmax_data = myfunc.softmax(tf.expand_dims(sample_data, axis=-1))
print('\nSoftmax data: ', softmax_data[:10])
print('\nMatched Softmax activation data: {0}'.format(tf.reduce_all(
    tf.equal(softmax(tf.expand_dims(sample_data, axis=-1)), softmax_data))))
myfunc.plot_sample_data(softmax_data, title='Softmax Data', c='g')

## Sigmoid Activation
sigmoid_data = myfunc.sigmoid(sample_data)
print('\nSigmoid data: ', sigmoid_data[:10])
print(sigmoid(sample_data))
print('\nMatched Sigmoid activation data: {0}'.format(tf.reduce_all(tf.equal(sigmoid(sample_data),
                                                                              sigmoid_data))))
myfunc.compare(tf.expand_dims(sigmoid(sample_data), axis=-1), 
               tf.expand_dims(sigmoid_data, axis=-1))
myfunc.plot_sample_data(sigmoid_data, title='Sigmoid Data', c='m')


