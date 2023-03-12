## Convolutional Neural Networks ##
'''
NOTE:
Computer vision is the practice of writing algorithms which can discover patterns
in visual data. Such as a camera of a self-driving car recognizing the car in front.
'''

import numpy as np
import matplotlib.image as mimg
import prepare_binary_data
import my_functions as myfunc

from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

print(f'\ntensorflow: {tf.__version__}')

'''
NOTE:
The images we are working with are from the Food101 dataset (101 different classes
of food): https://kaggle.com/dansbecker/food-101
However we are going to modify it to only use two classes (pizza and steak), to 
test on binary classification first.
'''
## Preparing binary data for binary classification
path = '../data/pizza_steak_images'
prepare_binary_data.prepare(path='../data/food-101/images', 
                            output_path=path,
                            datasets=['train', 'test'],
                            labels=['pizza', 'steak'])

## Plotting the data
# Getting the data labels
filenames = prepare_binary_data.get_labels('../data/food-101/meta', filename='train.json')
filenames = filenames['pizza'][:] + filenames['steak'][:]
filenames = [filename + '.jpg' for filename in filenames]
training_labels = [filename.split('/')[0] for filename in filenames]

myfunc.plot_images_sample(data=filenames,
                          labels=training_labels,
                          label_size=15,
                          dir_path=path + '/train',
                          read_from_dir=True)

## View one of the image shape
sample_img = mimg.imread(f'{path}/train/{filenames[0]}')
print(f'\nSample image shape: {sample_img.shape}\n') # Returns width, height, and colour channels

## Importing the whole dataset
train_path = path + '/train'
test_path = path + '/test'

train_datagen = ImageDataGenerator(rescale=1./225)
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary',
                                                 seed=42)
print(f'training_set length: {len(training_set)}') # total images found / batch size

validation_datagen = ImageDataGenerator(rescale=1./225)
validation_set = validation_datagen.flow_from_directory(test_path,
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary',
                                                        seed=42)
print(f'validation_set length: {len(validation_set)}') # total images found / batch size

## Building a CNN model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

'''
NOTE:
A way to build Convolutional Neural Networks:
*   Import all the images.
*   Preprocess the images.
*   Build a CNN to find patterns in out images.
*   Compile our CNN.
*   Fit the CNN to the training dataset.
'''

# Create the model
cnn_model = Sequential(name='cnn_model')

# Add layers to the model
cnn_model.add(Conv2D(filters=10, kernel_size=3, activation=relu, input_shape=(64, 64, 3)))
cnn_model.add(Conv2D(filters=10, kernel_size=3, activation=relu))
cnn_model.add(MaxPool2D(pool_size=2, padding='valid'))

cnn_model.add(Conv2D(filters=10, kernel_size=3, activation=relu))
cnn_model.add(Conv2D(filters=10, kernel_size=3, activation=relu))
cnn_model.add(MaxPool2D(pool_size=2))

cnn_model.add(Flatten())
cnn_model.add(Dense(units=128, activation=relu))
cnn_model.add(Dense(units=1, activation=sigmoid))

# Compile the model
cnn_model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Fit the model into the training set
print('\nFitting the CNN model.')
cnn_history = cnn_model.fit(training_set, 
                            epochs=10,
                            steps_per_epoch=len(training_set),
                            validation_data=validation_set, 
                            validation_steps=len(validation_set))

# Model summary
cnn_model.summary()

# Plot the accuracy and loss from the model history
myfunc.plot_model_history(cnn_history)

'''
NOTE:
You can think of trainable parameters as patterns a model can learn from the data.
Intuitively, you might think more is better. And in lots of cases, it is. But in
this case, the difference here is the two different styles of model we're using.
Where a series of Dense layers has a number of different learnable parameters connected
to each other and hence a higher number of possible learnable patterns, a convolutional
neural networks seek to sort out and learn the most important pattern in an image.
So even though these are less learnable parameters in our convolutional neural networks,
these are often more helpful in dechipering between different features in an image.
'''
