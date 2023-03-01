## Introduction to Neural Network Classification ##

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import my_functions as my_func

print('\ntensorflow: ', tf.__version__)

'''
NOTE:
Classification is where you try to classify something as one thing or another.
A few types of classificaiton problem:
1.  Binary Classification
2.  Multiclass Classification
3.  Multilabel Classification
'''

## Creating and viewing the data
from sklearn.datasets import make_circles

# Generate sample data
n_samples = 2000
X, y = make_circles(n_samples, noise=.05, random_state=42)

print('\nFeatures: \n', X[:10])
print('\nLabels: \n', y[:10])

# Create dataset from sample data
df = pd.DataFrame(np.concatenate((X, np.expand_dims(y, -1)), 1), 
                  columns=['X0', 'X1', 'Label'])
print('\nDataset: \n', df.head())

# Create dataset plot
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.title('Data plot')
plt.legend()
plt.show()

## Building a simple classification model
from tensorflow.keras import Sequential, layers, optimizers, losses

tf.random.set_seed(42)

# Creating a model
simple_model = Sequential(name='simple_model')

# Add layers to the model
simple_model.add(layers.Dense(10, name='input_layer'))
simple_model.add(layers.Dense(1, name='output_layer'))

# Compile the model
simple_model.compile(optimizer=optimizers.SGD(), 
                     loss=losses.BinaryCrossentropy(),
                     metrics=['accuracy'])

# Fit the model
simple_model.fit(X, y, epochs=10, verbose=0)

# Evaluate the model
print('\nsimple_model evaluation: \n')
simple_model.evaluate(X, y)

## Predict and visualize the prediction using the simple_model
'''
NOTE:
This part is about predicting and visualizing the prediction because the evaluation
from the simple_model is not looking good. So we are trying to find what's causing
this.
'''
my_func.plot_decision_boundary(simple_model, X, y)

## Building more complex classification model
tf.random.set_seed(42)

# Creating a model
better_model = Sequential(name='a_better_model')

# Add layers to the model
better_model.add(layers.Dense(10, activation='relu', name='input_layer'))
better_model.add(layers.Dense(10, activation='relu', name='hidden_layer'))
better_model.add(layers.Dense(1, activation='sigmoid', name='output_layer'))

# Compile the model
better_model.compile(optimizer=optimizers.Adam(learning_rate=.01),
                     loss=losses.BinaryCrossentropy(),
                     metrics=['accuracy'])

# Fit the model
better_model.fit(X, y, epochs=100, verbose=0)

# Evaluate the model
print('\nbetter_model evaluation: \n')
better_model.evaluate(X, y)

## Predict and visualize the prediction using the simple_model
my_func.plot_decision_boundary(better_model, X, y)

'''
NOTE:
The tricks we can use to improve our model:
*   Adding more layers
*   Increase the number of hidden units
*   Change the activation function
*   Change the optimization function
*   Change the learning rate
*   Fitting on more data
*   Fitting for longer
'''
