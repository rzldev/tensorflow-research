## Neural Network Regression ##

import tensorflow as tf

print('\ntensorflow: ', tf.__version__)

## Evaluating a model
'''
NOTE:
* Typical workflow when building a model:
build a model -> fit it -> evaluate a model -> tweak it -> evaluate it -> tweak it
-> fit it -> evaluate it -> tweak it -> fit it -> evaluate it -> ...

* Typical workflow when evaluate a model:
visualize it -> evaluate it -> visualize it -> evaluate it  -> visualize it -> ...

* It's a good idea to visualize:
1.  The data:
    What data are we working with? What does it look like?
2.  The model itself:
    What does our model look like?
3.  The training of the model:
    How does a model perform while it learns?
4.  The prediction of the model:
    How do the predictions of the model line up againts the ground truth (the 
    original label)?
'''

## Make a bigger dataset
X = tf.range(-100., 100., 4.)
y = tf.range(-250., 250., 10.)

## Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.numpy(), 
                                                    y.numpy(), 
                                                    test_size=.2, 
                                                    random_state=0)

X_train, X_test = tf.constant(X_train), tf.constant(X_test)
y_train, y_test = tf.constant(y_train), tf.constant(y_test)

print('\nX_train: ', X_train)
print('X_test: ', X_test)
print('y_train: ', y_train)
print('y_test: ', y_test)

'''
NOTE:
The 3 sets:
1.  Training set
    The model learns from this data, which is typically 70-80% of the total data 
    you have available.
2.  Validation set
    The model gets tuned on this data, which is typically 10-15% of the total
    data available.
3.  Test set
    The model gets evaluated on this data to test what it has learned, this set
    is typically 10-15% of the total data available.
'''

## Visualizing the data
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, c='blue', label='Training data')
plt.scatter(X_test, y_test, c='green', label='Test data')
plt.title('Data Plot')
plt.legend()
plt.show()

## Creating a model and show the summary
from tensorflow.keras import Sequential, layers, optimizers, losses
tf.random.set_seed(42)

# 1. Create a model
model = Sequential(name='simple_model')

# 2. Add input layer / first hidden layer
model.add(layers.Dense(100, activation='relu', input_shape=[1], name='input_layer'))

'''
NOTE:
* You need to specify the input_shape of the model to call the model summary.
* Sometime the model will automatically set, but sometime you will have to set
  it manually.
'''

# 3. Add more hidden layers
model.add(layers.Dense(100, activation='relu', name='hidden_layer_1'))

# 4. Add output layer
model.add(layers.Dense(1, name='output_layer'))

# 4. Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=.01),
              loss=losses.mae,
              metrics=['mae'])

# Call the model summary
model.build()
model.summary()

'''
NOTE:
* Total params: Total number of parameters in the model.
* Trainable parameters: These are the parameters (patterns) the model can 
                        update as it trains.
* Non-trainable parameters: These parameters aren't updated during training 
                            (This is typical when you bring in already learnt 
                            patterns or parameters from other models during
                            Transfer Learning)
'''

# 5. Fit the model
model.fit(X_train, y_train, epochs=100, verbose=0)

## Visualizing the model
from tensorflow.keras.utils import plot_model

plot_model(model=model, show_shapes=True)
# The output file will be exported on the same dir as the python file

## Visualizing the model predictions
import numpy as np

y_pred = model.predict(X_test)
print('\nComparing the model\'s prediction and the ground trurh:\n',
      np.concatenate((y_pred, tf.expand_dims(y_test, axis=-1)), axis=1))

plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, c='b', label='Training data')
plt.scatter(X_test, y_test, c='g', label='Test data')
plt.scatter(X_test, y_pred, c='r', label='Predicted data')
plt.title('Prediction Plot')
plt.legend()
plt.show()

## Evaluating the model predictions with regression evaluation metrics
from tensorflow.keras.metrics import mean_absolute_error as mae
from tensorflow.keras.metrics import  mean_squared_error as mse

'''
NOTE:
Depending on the problem you are working on, there will be different evaluation
metrics to evaluate the model performance. These are the two main evaluation
metrics to handler regression model problem:
1.  MAE (Mean Absolute Error)
    On average, how wrong is each of the model predictions.
2.  MSE (Mean Square Error)
    Square the average errors.
'''

# Evaluate the model on the test set
print('\nEvaluate the model on the test set.')
model.evaluate(X_test, y_test)

squeezed_y_pred = tf.squeeze(y_pred)
print('\nMean Absolute Error evaluation: \n', mae(y_test, squeezed_y_pred))
print('\nMean Squared Error evaluation: \n', mse(y_test, squeezed_y_pred))


