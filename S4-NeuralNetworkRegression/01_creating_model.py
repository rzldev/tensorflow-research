## Neural Network Regression ##

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print('\ntensorflow: ', tf.__version__)

## Creating sample data
X = np.arange(-7, 15, 3, dtype=np.float32) # Features
y = np.arange(3, 25, 3, dtype=np.float32) # Labels

print('''
Features: {0}
shape: {1}
datatype: {2}'''.format(X, X.shape, X.dtype))
print('''
Labels: {0}
shape: {1}
datatype: {2}'''.format(y, y.shape, y.dtype))

## Turn our NumPy arrays into tensors
X = tf.constant(X)
y = tf.constant(y)

print('\nX: ', X)
print('y: ', y)

## Visualize it
plt.scatter(X, y)
plt.show()

## Modeling with TensorFlow
tf.random.set_seed(42)

# Create a smaller model
# 1. Create a model with the Sequential API
small_model = tf.keras.Sequential()

# 2. Add input layer / first hidden layer
small_model.add(tf.keras.layers.Dense(1)) # input layer / first hidden layer

# 3. Compile the model
small_model.compile(optimizer=tf.keras.optimizers.SGD(), # sgd -> stochastic gradient descent
                    loss=tf.keras.losses.mae, # mae -> mean absolute error
                    metrics=['mae'])

# 4. Fit the model
small_model.fit(tf.expand_dims(X, axis=-1), y, epochs=10)

# Create a larger model
# 1. Create a model with the Sequential API
large_model = tf.keras.Sequential()

# 2. Add input layer / first hidden layer
large_model.add(tf.keras.layers.Dense(units=100, activation='relu'))

# 3. Add more hidden layers
large_model.add(tf.keras.layers.Dense(units=100, activation='relu'))

# 4. Add an output layer
large_model.add(tf.keras.layers.Dense(units=1))

# 4. Compile the model
large_model.compile(optimizer=tf.keras.optimizers.Adam(lr=.01),
                    loss=tf.keras.losses.mae,
                    metrics=['mae'])

# 5. Fit the model
large_model.fit(tf.expand_dims(X, axis=-1), y, epochs=100)

## Make a prediction using our model
print('\nSmaller model prediction for X = 17.0 is', small_model.predict([17.0]))
print('\nLarger model prediction for X = 17.0 is', large_model.predict([17.0]))

'''
NOTE:
The way we can use to improve our model.
1.  Creating a model: 
    Add more layers, increase the number of hidden neurons, change the activation
    function of each layer.
2.  Compiling a model:
    Change the optimization function or change the learning rate of the optimization
    function.
3.  Fitting a model:
    Fit a model for more epochs or on more data.
'''
