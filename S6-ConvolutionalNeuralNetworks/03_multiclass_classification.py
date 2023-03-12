## Convolutiona Neural Networks With Multi-Class Classification Problem ##

'''
NOTE:
Multi-class Classification plan:
1. Import, visualize & preprocess the data
2. Create a model (start with the baseline)
3. Fit the model (overfit it if possible, to make it works)
4. Evaluate the model
5. Adjust different hyperparameters and improve the model (try to beat the baseline
                                                           model and reduce overfitting)
6. Repeat until satisfied
'''

import numpy as np
import tensorflow as tf

import my_functions as myfunc

print(f'\ntensorflow: {tf.__version__}')

## Preparing multi-class classification data
import prepare_data as pd

path='../data/cnn_multiclass_classification'
class_names = ['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 
               'pizza', 'steak', 'hamburger', 'sushi', 'ramen', 'ice_cream']

pd.prepare(path='../data/food-101/images', output_path=path, labels=class_names)

# Visualize random data
filenames = pd.get_labels('../data/food-101', filename='train.json')
filenames = np.array([filenames[label] for label in filenames if label in class_names])
filenames = [label + '.jpg' for label in filenames.flatten()]
labels = [label.split('/')[0] for label in filenames]

myfunc.plot_images_sample(labels=labels, data=filenames,
                          label_size=15,
                          dir_path=f'{path}/train',
                          read_from_dir=True)
print()

## Load and preprocess the image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training set
train_datagen = ImageDataGenerator(rescale=1./225)
training_set = train_datagen.flow_from_directory(f'{path}/train',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')

# Test set
test_datagen = ImageDataGenerator(rescale=1./225)
test_set = test_datagen.flow_from_directory(f'{path}/test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')

# Build CNN multiclass classification model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# Create the model
model_1 = Sequential(name='cnn_model_1')

# Add layers to the model
model_1.add(Conv2D(filters=10, 
                   kernel_size=3, 
                   activation=relu, 
                   input_shape=(64, 64, 3), 
                   name='input_layer'))
model_1.add(Conv2D(filters=10, kernel_size=3, activation=relu))
model_1.add(MaxPool2D())
model_1.add(Conv2D(filters=10, kernel_size=3, activation=relu))
model_1.add(Conv2D(filters=10, kernel_size=3, activation=relu))
model_1.add(MaxPool2D())
model_1.add(Flatten())
model_1.add(Dense(units=len(class_names), activation=softmax, name='output_layer'))

# Compile the model
model_1.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

# Model summary
print('\nModel Summary:')
model_1.summary()

# Fit the model into the training set
print('\nFitting the model into the training set')
model_1_history = model_1.fit(training_set,
                              epochs=5,
                              steps_per_epoch=len(training_set),
                              validation_data=test_set,
                              validation_steps=len(test_set))

# Evaluate the model
print('\nEvaluating model_1')
model_1.evaluate(test_set)
myfunc.plot_model_history(model_1_history)

'''
NOTE:
If your val_loss and training_loss or val_accuracy and training_accurcay plots going
to a different direction means that you model is overfitting... in other words,
it's getting a great results on the training data but fails to generalize well to
unseen data and performs poorly on the test set.

A way to fix overfitting:
*   Get more data.
    Having more data gives a model more opportunity to learn diverse patterns.
*   Simplify the model.
    If the current model is overfitting the data, it maybe too complicated of a 
    model, one way to simplify a model is to: reduce # of layers or reduce # of
    units in layers.
*   Use data augmentation.
    Data augmentation manipulates the training data in such a way to add more
    diversity to it (without altering the original data).
*   Use transfer learning.
    Transfer learning leverages the patterns another model has learned on similar
    data to your own and allows you to you those patterns on your own dataset.
'''

## Build a better model from the baseline model

# Implement data augmentation on the training set
augmented_train_datagen = ImageDataGenerator(rescale=1./225,
                                             zoom_range=.2,
                                             shear_range=.2,
                                             rotation_range=.2,
                                             width_shift_range=.2,
                                             height_shift_range=.2,
                                             horizontal_flip=True)
augmented_training_set = augmented_train_datagen.flow_from_directory(f'{path}/train',
                                                                     target_size=(64, 64),
                                                                     batch_size=32,
                                                                     class_mode='categorical')

# Create the CNN model
model_2 = Sequential(name='model_2')

# Add layers to the model
model_2.add(Conv2D(filters=32, kernel_size=3, activation=relu, 
                   input_shape=(64, 64, 3), name='input_layer'))
model_2.add(Conv2D(filters=32, kernel_size=3, activation=relu))
model_2.add(MaxPool2D(pool_size=2, strides=2))

model_2.add(Conv2D(filters=32, kernel_size=3, activation=relu))
model_2.add(Conv2D(filters=32, kernel_size=3, activation=relu))
model_2.add(MaxPool2D(pool_size=2, strides=2))

model_2.add(Conv2D(filters=32, kernel_size=3, activation=relu))
model_2.add(Conv2D(filters=32, kernel_size=3, activation=relu))
model_2.add(MaxPool2D(pool_size=2, strides=2))

model_2.add(Flatten())
model_2.add(Dense(units=128, activation=relu))
model_2.add(Dense(units=128, activation=relu))
model_2.add(Dense(units=len(class_names), activation=softmax, name='output_layer'))

# Compile the model
model_2.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

# Model summary
print('\nModel Summary:')
model_2.summary()

# Fit the model into the augmented training set
print('\nFitting the model into the augmented training set')
model_2_history = model_2.fit(augmented_training_set,
                              epochs=25,
                              steps_per_epoch=len(augmented_training_set),
                              validation_data=test_set,
                              validation_steps=len(test_set))

# Evaluate the model
print('\nEvaluating model_2')
model_2.evaluate(test_set)
myfunc.plot_model_history(model_2_history)

## Making predictions
print('\nMaking predictions')
myfunc.predict_and_visualize(model_2, 
                             labels=class_names, 
                             img_path='../data/03-pizza-dad.jpeg')
myfunc.predict_and_visualize(model_2, 
                             labels=class_names, 
                             img_path='../data/03-hamburger.jpeg')
myfunc.predict_and_visualize(model_2, 
                             labels=class_names, 
                             img_path='../data/03-sushi.jpeg')
myfunc.predict_and_visualize(model_2, 
                             labels=class_names, 
                             img_path='../data/03-steak.jpeg')

## Saving and loading the model
from tensorflow.keras.models import load_model

# Save a model
model_2.save('trained_model_2')

# Loaded in the trained model
loaded_model_2 = load_model('trained_model_2')

# Compare the loaded model and the existing model
print('\nCompare model_2 and loaded_model_2')
model_2.evaluate(test_set)
loaded_model_2.evaluate(test_set)
