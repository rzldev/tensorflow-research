## Evaluating and Improving Neural Network Classification ##

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Tensorflow
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import LearningRateScheduler

print('\ntensorflow: ', tf.__version__)

# Custom functions
import my_functions as myfunc

## Create the sample data
from sklearn.datasets import make_circles

# Generate sample data
n_samples = 1000
X, y = make_circles(n_samples, noise=.05, random_state=42)
print('\nFeatures: \n', X[:10])
print('\nLabels: \n', y[:10])

# Create dataset from sample data
df = pd.DataFrame(np.concatenate((X, np.expand_dims(y, -1)), 1), 
                  columns=['X0', 'X1', 'Label'])
print('\nDataset: \n', df.head())

X = df.drop('Label', axis=1)
y = df.iloc[:, -1]

## Splitting the data into training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

print('\nX_train: \n{0} \n\nX_test: \n{1} \n\ny_train: \n{2} \n\ny_test: \n{3}'
      .format(X_train.head(), X_test.head(), y_train.head(), y_test.head()))

# Create training set plot
myfunc.create_data_plot(X_train.X0, X_train.X1, y_train, title='Training Set Data Plot')

# Create test set plot
myfunc.create_data_plot(X_test.X0, X_test.X1, y_test, title='Test Set Data Plot')

''' Create a simple classification model '''

simple_model_layers = [
    Dense(10, activation=relu, name='input_layer'),
    Dense(10, activation=relu, name='hidden_layer')
    ]
simple_model, simple_model_history = myfunc.create_model(X_train, y_train, 
                                                         name='simple_model',
                                                         model_layers=simple_model_layers,
                                                         output_activation=sigmoid,
                                                         optimizer=Adam(learning_rate=.01),
                                                         loss=BinaryCrossentropy(),
                                                         metrics=['accuracy'],
                                                         epochs=50,
                                                         verbose=0)

# Plot the loss (or training) curves
simple_history_df = pd.DataFrame(simple_model_history.history)
myfunc.plot_sample_data(simple_history_df, title='Simple Model History Plot')

'''
NOTE:
For many problems, the loss function going down means the model is improving (the
predictions it's making are getting closer to the ground truth labels).
'''

''' Create a better classification model '''

# Create the model
better_model = Sequential(name='better_model')

# Add layers
better_model.add(Dense(10, activation=relu, name='input_layer'))
better_model.add(Dense(10, activation=relu, name='hidden_layer'))
better_model.add(Dense(1, activation=sigmoid, name='output_layer'))

# Compile the model
better_model.compile(optimizer=Adam(learning_rate=.001),
                     loss=BinaryCrossentropy(),
                     metrics=['accuracy'])

# Create learning rate callback
lr_scheduler = LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))

# Fit the model on the dataset
better_model_history = better_model.fit(X_train, y_train, 
                                        epochs=60, verbose=0,
                                        callbacks=[lr_scheduler])

# Plot the loss (or training) curves
better_history_df = pd.DataFrame(better_model_history.history)
myfunc.plot_sample_data(better_history_df, 
                        title='Better Model History Plot', 
                        xlabel='epochs')

## Plot the learning rate versus the loss
lrs = 1e-4 * (10 ** (tf.range(60)/20))
plt.figure(figsize=(8, 4))
plt.semilogx(lrs, better_history_df.loss)
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate vs Loss')
plt.legend()
plt.show()

''' 
Create way better classification model based on better_model learning_rate and
loss analysis 
'''

# Create the model
better_model_2_layers = [
    Dense(10, activation=relu, name='input_layer'),
    Dense(10, activation=relu, name='hidden_layer')
    ]
better_model_2, better_model_2_history = myfunc.create_model(
    X_train, y_train, name='better_model_2', model_layers=better_model_2_layers,
    output_activation=sigmoid, optimizer=Adam(learning_rate=.02), 
    loss=BinaryCrossentropy(), metrics=['accuracy'], epochs=25, verbose=0)

# Plot the loss (or training) curves
better_2_history_df = pd.DataFrame(better_model_2_history.history)
myfunc.plot_sample_data(better_2_history_df, 
                        title='Better Model 2 History Plot', 
                        xlabel='epochs')

## Evaluating the models
myfunc.evaluate_models(X_test, y_test, simple_model)
myfunc.evaluate_models(X_test, y_test, better_model)
myfunc.evaluate_models(X_test, y_test, better_model_2)

## Decision boundary plots
myfunc.plot_decision_boundary(simple_model, X_test.values, y_test, 
                              title='Simple Model Test Data Plot', verbose=0)
myfunc.plot_decision_boundary(better_model, X_test.values, y_test,
                              title='Better Model Test Data Plot', verbose=0)
myfunc.plot_decision_boundary(better_model_2, X_test.values, y_test,
                              title="Better Model 2 Test Data Plot", verbose=0)

## Making predictions
y_test_rank_2 = np.expand_dims(y_test.values, axis=-1)

simple_y_pred = simple_model.predict(X_test)
print('\ny_test vs predictions from simple classification model: \n', 
      np.concatenate((y_test_rank_2[:10], np.round(simple_y_pred[:10])), axis=1))

better_y_pred = better_model_2.predict(X_test)
print('\ny_test vs predictions from better calssification model (2): \n',
      np.concatenate((y_test_rank_2[:10], np.round(better_y_pred[:10])), axis=1))

## More classification evaluation methods
'''
NOTE:
Alongside visualizing our models results as much as possible, there are handful
of other classification evaluation methods & metrics which can be used.
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report
'''

from sklearn.metrics import accuracy_score, confusion_matrix

print('\nSimple model accuracy score: ', accuracy_score(y_test_rank_2, 
                                                       np.round(simple_y_pred)))
print('Simple model confusion matrix: \n', confusion_matrix(y_test_rank_2, 
                                                              np.round(simple_y_pred)))
# Plot confusion matrix for simple_model
myfunc.plot_confusion_matrix(y_test_rank_2, np.round(simple_y_pred), 
                             title='Simple Model Confusion Matrix')


print('\nBetter model (2) accuracy score: ', accuracy_score(y_test_rank_2, 
                                                       np.round(better_y_pred)))
print('Better model (2) confusion matrix: \n', confusion_matrix(y_test_rank_2, 
                                                              np.round(better_y_pred)))
# Plot confusion matrix for better_model_2
myfunc.plot_confusion_matrix(y_test_rank_2, np.round(better_y_pred),
                             title='Better Model (2) Confusion Matrix')
