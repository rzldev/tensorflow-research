## Multi-Class Classification ##

'''
Working with bigger sample data (Multi-Class Classification). When you have more
than two classes as an options, it's known as multi-class classification.
* This means you will have 3 different classes or more (there is no limit)

To practice multi-class classification we are going to build a neural network
to classify images of different clothing from tensorflow datasets.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rand
import tensorflow as tf
import my_functions as myfunc

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.callbacks import LearningRateScheduler

print(f'\ntensorflow: {tf.__version__}')

## Importing the dataset
from tensorflow.keras.datasets import fashion_mnist

(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

print(f'''\nTraining feature: {train_data[0].shape}\n{train_data[0]} 
      \nTraining label: {train_labels[0].shape}\n{train_labels[0]}
      \nTest feature: {test_data[0].shape}\n{test_data[0]} 
      \nTest label: {test_labels[0].shape}\n{test_labels[0]}\n''')
      
class_names = ('T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
print (len(class_names), max(train_labels), max(test_labels))
      
plt.figure(figsize=(6, 6))
for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    rand_index = rand.choice(range(len(train_data)))
    plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
    plt.title(class_names[train_labels[rand_index]])
    ax.title.set_size(16)
    plt.axis(False)
plt.show()

'''
Create a simple dataset.
'''

tf.random.set_seed(42)

# Create the model
simple_model = Sequential(name='simple_model')

# Add layers to the model
simple_model.add(Flatten(input_shape=(28, 28))) # The data need to be flattened
simple_model.add(Dense(20, activation=relu, name='input_layer'))
simple_model.add(Dense(20, activation=relu, name='hidden_layer'))
simple_model.add(Dense(10, activation=softmax, name='output_layer'))

# Compile the model
# simple_model.compile(optimizer=Adam(lr=.0001),
#                      loss=CategoricalCrossentropy(),
#                      metrics=['accuracy'])
simple_model.compile(optimizer=Adam(lr=.001), 
                      loss=SparseCategoricalCrossentropy(), 
                      metrics=['accuracy'])

# Learning rate callback
lr_scheduler = LearningRateScheduler(lambda epoch: 1e-3 * (10 ** (epoch/20)))

# Fitting the model into the dataset
print('\nFitting ', simple_model.name)
# simple_model.fit(train_data, tf.one_hot(train_labels, depth=10), epochs=25, verbose=1)
simple_model_history = simple_model.fit(train_data, 
                                        train_labels, 
                                        epochs=100,
                                        validation_data=(test_data, test_labels),
                                        verbose=1)

# Plot the learning rate decay curve
myfunc.plot_learning_rate(lrs=1e-3 * (10 ** (tf.range(100)/20)), 
                          history_loss=pd.DataFrame(simple_model_history.history).loss)

'''
NOTE: Building Multi-Class Classification Model
For multi-class classification model, we can use a similar architecture to our 
binary classifiers, however we have to tweak a few things:
*   Input shape => Set the same as the input shape from the data (have to flatten 
    it sometimes)
*   Output shape => Set the same as how many label class you have for your data
*   Loss function => You can either use SparseCategoricalCrossentropy or 
    CategoricalCrossentropy. 
    *   If your label data are in integer form you are suggested to use 
        SparseCategoricalCrossentropy. 
    *   Someimes you have to one-hot encoded the data if you want to use
        CategoricalCrossentropy.
*   Output layer activation: Softmax (not Sigmoid).
'''

'''
Create a better dataset.
'''

## Data preprocessing (Normalization)
print(f'\nMin value of train_data: {train_data.min()} \nMax value of train_data: {train_data.max()}')
print(f'\nMin value of test_data: {test_data.min()} \nMax value of test_data: {test_data.max()}')

train_data_norm = train_data / train_data.max()
test_data_norm = test_data / test_data.max()

## Create the model with normalized data
tf.random.set_seed(42)

# Create the model
better_model = Sequential(name='better_model')

# Add layers to the model
better_model.add(Flatten(input_shape=(28, 28))) # The data need to be flattened
better_model.add(Dense(20, activation=relu, name='input_layer'))
better_model.add(Dense(20, activation=relu, name='hidden_layer'))
better_model.add(Dense(10, activation=softmax, name='output_layer'))

# Compile the model
# better_model.compile(optimizer=Adam(lr=.002),
#                      loss=CategoricalCrossentropy(),
#                      metrics=['accuracy'])
better_model.compile(optimizer=Adam(lr=.002), 
                      loss=SparseCategoricalCrossentropy(), 
                      metrics=['accuracy'])

# Fitting the model into the dataset
print('\nFitting ', better_model.name)
# better_model.fit(train_data_norm, tf.one_hot(train_labels, depth=10), epochs=25, verbose=1)
better_model_history = better_model.fit(train_data_norm, 
                                        train_labels, 
                                        epochs=25, 
                                        validation_data=(test_data_norm, test_labels),
                                        verbose=0)

'''
NOTE:
Neural networks tend to work better in numerical form as well as scaled/normalized
(numbers between 0 & 1).
'''

## Comparing the models
print('\nEvaluate simple multiclass classification model:\n')
# simple_model.evaluate(test_data, tf.one_hot(test_labels, depth=10))
simple_model.evaluate(test_data, test_labels)

print('\nEvaluate better multiclass classification model:\n')
# better_model.evaluate(test_data, tf.one_hot(test_labels_norm, depth=10))
better_model.evaluate(test_data_norm, test_labels)

myfunc.plot_sample_data(pd.DataFrame(simple_model_history.history),
                        title='Simple Model History Plot',
                        xlabel='Epochs')

myfunc.plot_sample_data(pd.DataFrame(better_model_history.history),
                        title='Better Model History Plot',
                        xlabel='Epochs')

## Evaluating better_model with confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

y_probs = better_model.predict(test_data_norm)
y_preds = y_probs.argmax(axis=1)

print(f'Accuracy score: {accuracy_score(test_labels, y_preds)}')
print(f'Confusion matrix: {confusion_matrix(test_labels, y_preds)}')

myfunc.plot_confusion_matrix(test_labels, y_preds, figsize=(20, 15), 
                             title='Model Confusion Matrix Plot',
                             title_size=24, label_size=20, text_size=12)

## What patterns are our model learning?

# Find the layers of the better_model
print(f'\nbetter_model layers: \n{better_model.layers}')

# Extract a particular layer from better_model
first_hidden_layer = better_model.layers[1]
print(f'\nFirst hidden layer of better_model: \n{first_hidden_layer}')

# Get the patterns from first hidden layer
weights, biases = first_hidden_layer.get_weights()
print(f'\nweights: {weights.shape} \n{weights}')
print(f'\nbiases: {biases.shape} \n{biases}')

'''
NOTE:
*   Every neuron has a bias vector. Each of these is paired with a weights matrix.
*   The bias vector get initialized as zeros (at least in the case of TensorFlow
    Dense layer).
*   The bias vector dictates how much the patterns within the corresponding weights
    matrix should influence the next layer.
'''

# Check model summary
from tensorflow.keras.utils import plot_model

better_model.summary()
plot_model(better_model, show_shapes=True)
























