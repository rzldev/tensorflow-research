## Extras on Neural Network Regression ##

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers, losses, models

print('\ntensorflow:', tf.__version__)

## Useful methods
def create_model(X_train, y_train, model_layers=[], name=None, epochs=1, optimizer=optimizers.SGD(),
                 loss=losses.mae, metrics=['mae']):
    model = Sequential(name=name)
    for layer in model_layers:
        model.add(layer)
    model.add(layers.Dense(1, name='output_layer'))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    print('\n{0} summary: '.format(name))
    model.build()
    model.summary()
    
    print('\nFitting {0}'.format(name))
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    
    return model

## Preparing data
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

## Save TensorFlow model
model = create_model(X_train, y_train, model_layers=[
    layers.Dense(100, activation='relu', input_shape=[1], name='input_layer'),
    layers.Dense(100, activation='relu', name='hidden_layer_1'),
    ], name='model', epochs=100)

'''
NOTE:
Saving our models allow us to use them outside the place they are trained such as
web or mobile applications. These are two main formats we can use to save our models:
1.  The SavedModel format
2.  The HDF5 format
'''

# Save model using SavedModel format 
model.save('temp/model_with_SavedModel_format')

# Save model using HDF5 format
model.save('temp/model_with_HDF5_format.h5')

## Load saved model
loaded_model = models.load_model('temp/model_with_SavedModel_format')
loaded_h5_model = models.load_model('temp/model_with_HDF5_format.h5')

# Model summaries
print('\nSavedModel model summary: \n', loaded_model.summary())
print('\nHDF5 model summary: \n', loaded_h5_model.summary())

# Compare the predicted value to check if loaded model is the same as the saved one
print('\nComparing SavedModel model with the one created before: \n', 
      np.concatenate((model.predict(X_test), loaded_model.predict(X_test)), axis=1))
print('\nComparing HDF5 model with the one created before: \n', 
      np.concatenate((model.predict(X_test), loaded_model.predict(X_test)), axis=1))


