## Dive Deeper Into Convolutional Neural Networks With Binary Classification Problem ##

'''
NOTE:
Breaking down Binary Classification steps:
1.  Visualize the data (trying to understand the data)
2.  Preprocess the data (prepared it for our model, the main steps here is 
    scaling/normalizing & turning our data into batches)
3.  Create a baseline model
4.  Fit the model
5.  Evaluate the model
6.  Adjust different parameters and improve the model (try to beat the baseline)
7.  Repeat until satisfied
'''

import tensorflow as tf
import my_functions as myfunc

print(f'/ntensorflow: {tf.__version__}')

## Preparing the binary classification data
import prepare_data as pd

path = '../data/pizza_steak_images'
class_names=['pizza', 'steak']
pd.prepare(path='../data/food-101/images', output_path=path, labels=class_names)

## Visualizing random sample from the dataset
filenames = pd.get_labels('../data/food-101', filename='train.json')
filenames = filenames['pizza'][:] + filenames['steak'][:]
filenames = [file + '.jpg' for file in filenames]
labels = [file.split('/')[0] for file in filenames]

myfunc.plot_images_sample(labels=labels, data=filenames, 
                          label_size=15,
                          dir_path=f'{path}/train/', 
                          read_from_dir=True)
print()

## Import the training set and test set
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training set
train_path = path + '/train'
train_datagen = ImageDataGenerator(rescale=1./225) # Preprocess the data
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')
print(f'training_set size: {len(training_set)}\n')

# Test set
test_path = path + '/test'
test_datagen = ImageDataGenerator(rescale=1./225) # Preprocess the data
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
print(f'test_set size: {len(test_set)}')

# Check the data shape
images, labels = training_set.next() # Get a sample of next training data batch
print(f'\nimages size: {len(images)} \nlabels size: {len(labels)}')
print(f'sample image shape: {images[0].shape}') # Will be used for model input shape
print(f'labels: {labels[:10]}')


'''
NOTE:
A batch is a small subset of data. Rather than look at all ~10.000 images at one
time, a model might only look at 32 at a time. It does this for a couple of reasons:
1.  10.000 (or more) images might not fit into the memory of your processor (GPU).
2.  Trying to learn the patterns in 10.000 (or more) images in one hit could result
    in model not being able to learn very well.
    
Why 32? Because more than 32 can be bad for the model to learn the data.
'''

## Building a simple CNN model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Create the model
model_1 = Sequential(name='cnn_model_1')

# Add layers to the model
model_1.add(Conv2D(filters=10, 
                   kernel_size=3, 
                   strides=1, 
                   padding='valid',
                   activation=relu, 
                   input_shape=(64, 64, 3), 
                   name='input_layer'))
model_1.add(Conv2D(filters=10, kernel_size=3, activation=relu))
model_1.add(Conv2D(filters=10, kernel_size=3, activation=relu))
model_1.add(Flatten())
model_1.add(Dense(units=1, activation=sigmoid, name='output_layer'))

'''
NOTE:
* filters:
    the number of sliding window going acorss an input (higher = more complex model)
* kernel_size:
    the size of the sliding window going across an input
* strides:
    the size of step of the sliding window takes across an input
* padding:
    if "same", output shape is same as input shape. if "valid", output shape gets
    compressed
'''

# Compile the model
model_1.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Model summary
print('\nModel Summary:')
model_1.summary()

# Fitting the model into the dataset
epoch_steps = len(training_set)
validation_steps = len(test_set)

print('\nFitting the model')
model_1_history = model_1.fit(training_set, 
                              epochs=10,
                              steps_per_epoch=epoch_steps,
                              validation_data=test_set,
                              validation_steps=validation_steps)

# Plot the accuracy and loss from model history
myfunc.plot_model_history(model_1_history)

'''
NOTE:
When a model's validation loss start to increase, it's likely that the model is 
overfitting the training dataset. This means, it's learning the patterns in the
dataset too well and thus the model's ability to generalize the unseen data will
be diminished.
'''

## Building a better 
'''
NOTE:
Fitting a machine learning model comes in 3 steps:
1.  Create a baseline
2.  Beat the baseline by overfitting a larger model
3.  Reduce overfitting

Ways to induce overfitting:
*   Increase the number of conv layers
*   Increase the number of conv filters
*   Add another dense layer to the output of our flattened layer

Ways to reduce overfitting:
*   Add data augmentation
*   Add regularization layers (such as MaxPool2D)
*   Add more data
'''

# Create the model
model_2 = Sequential(name='cnn_model_2')

# Add layers to the model
model_2.add(Conv2D(filters=10,
                   kernel_size=3,
                   activation=relu,
                   input_shape=(64, 64, 3),
                   name='input_layer'))
model_2.add(MaxPool2D())
model_2.add(Conv2D(filters=10, kernel_size=3, activation=relu))
model_2.add(MaxPool2D())
model_2.add(Conv2D(filters=10, kernel_size=3, activation=relu))
model_2.add(Flatten())
model_2.add(Dense(units=1, activation=sigmoid, name='output_layer'))

# Compile the model
model_2.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Model summary
print('\nModel Summary:')
model_2.summary()

# Fitting the model into the dataset
print('\nFitting the model')
model_2_history = model_2.fit(training_set,
                              epochs=10,
                              steps_per_epoch=epoch_steps,
                              validation_data=test_set,
                              validation_steps=validation_steps)

# Plot the accuracy and loss from model history
myfunc.plot_model_history(model_2_history)

## Reducing overfitting with data augmentation
train_datagen_augmented = ImageDataGenerator(rescale=1./225,
                                             zoom_range=.2,
                                             shear_range=.2,
                                             rotation_range=.2,
                                             width_shift_range=.2,
                                             height_shift_range=.2,
                                             horizontal_flip=True)
augmented_training_set = train_datagen_augmented.flow_from_directory(train_path,
                                                                     target_size=(64, 64),
                                                                     batch_size=32,
                                                                     class_mode='binary')
print(f'\naugmented_training_set size: {len(augmented_training_set)}')

'''
NOTE:
What is data augmentation?
->  The process of altering the training data, leading it to have more diversity
    and in turn allowing the models to learn more generalize (hopefully) patterns.
    Altering might mean adjusting the rotation of an image, flipping it, cropping
    it, or something similar.

* rotation_range: how much do you want to rotate the image
* zoom_range: randomly zoom in on the image
* shear_range: how much do you want to shear the image
* width_shift_range: move your image around on the x-axis
* height_shift_range: move your image around on the y-axis
* horizontal_flip: do you want to flip the image

Data augmentation usually only performed on the training data. Using ImageDataGenerator
built-in data augmentation parameters our images as left as in the directories 
but are modified as they are loaded into the model.
'''

# Visualizing the augmented image
myfunc.plot_augmented_data(augmented_training_set, training_set, label_size=15)

## Build a CNN model and fitting it into the augmented data

# Create a model
model_3 = Sequential(name='cnn_model_3')

# Add layers to the model
model_3.add(Conv2D(filters=32,
                   kernel_size=3,
                   activation=relu,
                   input_shape=(64, 64, 3),
                   name='input_layer'))
model_3.add(Conv2D(filters=32, kernel_size=3, activation=relu))
model_3.add(MaxPool2D(pool_size=2))
model_3.add(Conv2D(filters=32, kernel_size=3, activation=relu))
model_3.add(Conv2D(filters=32, kernel_size=3, activation=relu))
model_3.add(MaxPool2D(pool_size=2))
model_3.add(Conv2D(filters=32, kernel_size=3, activation=relu))
model_3.add(Conv2D(filters=32, kernel_size=3, activation=relu))
model_3.add(MaxPool2D(pool_size=2))
model_3.add(Flatten())
model_3.add(Dense(units=128, activation=relu))
model_3.add(Dense(units=1, activation=sigmoid, name='output_layer'))

# Compile the model
model_3.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Model summary
print('\nModel Summary:')
model_3.summary()

# Fitting the model into the dataset
print('\nFitting the model')
model_3_history = model_3.fit(augmented_training_set,
                              epochs=20,
                              steps_per_epoch=len(augmented_training_set),
                              validation_data=test_set,
                              validation_steps=validation_steps)

# Plot the accuracy and loss from model history
myfunc.plot_model_history(model_3_history)

## Making predictions with custom data
import matplotlib.pyplot as plt
import matplotlib.image as mimg

# Importing the image
steak_img_path = '../data/03-steak.jpeg'
steak_img = mimg.imread(steak_img_path)
plt.imshow(steak_img)
plt.title('Steak')
plt.axis(False)
plt.show()

# Preprocess the custom data
steak_img = myfunc.load_and_preprocess_image(steak_img_path, target_size=(64, 64))

# Make the prediction
steak_img_pred = model_3.predict(tf.expand_dims(steak_img, axis=0))
print(f'\nPrediction: {class_names[int(tf.round(steak_img_pred))]}')

# Making another prediction
myfunc.predict_and_visualize(model_3, 
                             labels=class_names, 
                             img_path='../data/03-pizza-dad.jpeg')
