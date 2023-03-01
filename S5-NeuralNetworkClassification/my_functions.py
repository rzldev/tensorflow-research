## All of My Useful Functions ##

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix

# Tensorlofw
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mae
from tensorflow.keras.callbacks import LearningRateScheduler

def create_model(X_train, y_train, model_layers=[], output_activation='',name=None, 
                 epochs=1, optimizer=SGD(), loss=mae, metrics=['mae'], verbose=1):
    '''
    Create ANN model.
    '''
    model = Sequential(name=name)
    for layer in model_layers:
        model.add(layer)
    model.add(Dense(1, activation=output_activation, name='output_layer'))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # print('\n{0} summary: '.format(name))
    # model.build()
    # model.summary()
    
    print('\nFitting {0}'.format(name))
    history = model.fit(X_train, y_train, epochs=epochs, verbose=verbose)
    
    return model, history

'''
NOTE:
This function will:
*   Take in a trained model, features (X) and labels (y)
*   Create a meshgrid of the different X values
*   Make Prediction across the meshgrid
*   Plot the prediction as well as a lines between zones (where each unique class
    falls)
'''
def plot_decision_boundary(model, X, y, title='', verbose=1):
    print('\nCreating "{0}"'.format(title))
    '''
    Plots the decision boundary created by a model predicting on X.
    '''
    # Define the axis boundaries of the plot and create meshgrid.
    X_min, X_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    if verbose == 1:
        print('\nX_min: {0} \nX_max: {1} \ny_min: {2} \ny_max: {3}'.format(X_min, X_max,
                                                                           y_min, y_max))
    
    XX, yy = np.meshgrid(np.linspace(X_min, X_max, 100),
                         np.linspace(y_min, y_max, 100))
    # print('\nXX: \n{0} \n\nyy: \n{1}'.format(XX[:2], yy[:2]))
    
    # Create X value (making predictions on these)
    X_in = np.c_[XX.ravel(), yy.ravel()] # stack 2D array together
    
    # Make predictions
    y_pred = model.predict(X_in)
    
    # Check for multiclass
    if len(y_pred[0]) > 1:
        if verbose == 1:
            print('\nDoing multiclass classification.')
        # Reshape the predicitons to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(XX.shape)
    else:
        if verbose == 1:
            print('\nDoing binary classification.')
        y_pred = np.round(y_pred).reshape(XX.shape)
        
    # Plot the decision boundary
    plt.contourf(XX, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(XX.min(), XX.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.show()
    
    '''
    NOTE:
    * meshgrid: 
        used to create rectangular grid out of two-given one-dimensional arrays representing 
        the cartesian index or matrix index.
    '''
    
def create_data_plot(X0, X1, y, title=''):
    '''
    Create simple data plot.
    '''
    plt.scatter(X0, X1, c=y, cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.legend()
    plt.show()
    
def evaluate_models(X_test, y_test, *models):
    '''
    Evaluate the models.
    '''
    for model in models:
        print('\n{0} evaluation: \n'.format(model.name))
        model.evaluate(X_test, y_test)
        
def plot_confusion_matrix(y_test, y_pred, title='Confusion Matrix', figsize=(5, 5),
                          label_size=11, title_size=14, text_size=12):
    '''
    Plot confusion matrix.
    '''
    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize our confusion matrix
    n_classes = cm.shape[0]
    
    # Pretify it
    fig, ax = plt.subplots(figsize=figsize)
    # Create a matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    # Create classes
    classes = False
    
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    
    # Label the axes
    ax.set(title=title, xlabel='Predicted Label', ylabel='True Label',
           xticks=np.arange(n_classes), yticks=np.arange(n_classes),
           xticklabels=labels, yticklabels=labels)
    
    # Set x-axis labels to bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    
    # Adjust label size
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.title.set_size(title_size)
    
    # Set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.
    
    # Plot the text on each cell
    for i, j in itertools.product(range(n_classes), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)',
                 horizontalalignment='center', size=text_size,
                 color='white' if cm[i, j] > threshold else 'black')
    
def plot_learning_rate(lrs, history_loss):
    '''
    Plot the learning rate decay curve.
    '''
    plt.figure(figsize=(8, 4))
    plt.semilogx(lrs, history_loss)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs Loss')
    plt.legend()
    plt.show()

    
'''
NOTE:
Most of the description and formulas can be found in tensorflow.org
'''
from tensorflow.math import maximum, exp

def linear(data):
    '''
    Linear activation will return the input, unmodified.
    '''
    return data

def relu(data):
    '''
    With default values, this returns the standard ReLU activation: max(x, 0), 
    the element-wise maximum of 0 and the input tensor.
    '''
    return maximum(0, data)

def sigmoid(data):
    '''
    For small values (< -5), sigmoid returns a value close to zero, and for large 
    values (> 5) the result of the function gets close to 1.
    '''
    return (1 / (1 + exp(-data)))

def softmax(data):
    '''
    Softmax converts a vector of values to a probability distribution. The elements 
    of the output vector are in range (0, 1) and sum to 1.
    '''
    return tf.map_fn(fn=(lambda x: exp(x) / tf.reduce_sum(exp(x))), elems=data)

def plot_sample_data(data, title='A Plot', c='', xlabel='', ylabel=''):
    print('\nCreating "{0}"'.format(title))
    if isinstance(data, pd.DataFrame):
        data.plot()
    else:
        plt.plot(data, c=c, cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def compare(arr1, arr2, title=''):
    print(title, np.concatenate((arr1, arr2), axis=1))




