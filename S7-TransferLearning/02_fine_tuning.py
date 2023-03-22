## Transfer Learning - Fine Tuning ##

import os
import tensorflow as tf
import my_functions as myfunc
import prepare_data as prep

print(f'\ntensorflow: {tf.__version__}')

OG_PATH = '../data/food-101/images'
PATH = '../data/food_101_10_percent'

TARGET_SIZE = (224, 224)
BATCH_SIZE=32

CLASS_NAMES = ['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 
               'pizza', 'steak', 'hamburger', 'sushi', 'ramen', 'ice_cream']

## Preapring the data
prep.prepare(OG_PATH, PATH, 
             datasets=['train', 'test'],
             labels=CLASS_NAMES,
             train_data_size=.1)

# Visualize the imported data
file_dirs = os.listdir(PATH + '/train')
filenames = []
for fdir in file_dirs:
    for filename in os.listdir(f'{PATH}/train/{fdir}'):
        filenames.append(f'{fdir}/{filename}')
myfunc.plot_sample_images(filenames,
                          label_size=15,
                          dir_path=PATH + '/train',
                          read_from_dir=True)
print()

## Importing the dataset
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Training set
train_dir = PATH + '/train'
training_set = image_dataset_from_directory(directory=train_dir, 
                                            image_size=TARGET_SIZE,
                                            batch_size=BATCH_SIZE,
                                            label_mode='categorical')
print('training_set: ', training_set, '\n')

# Test set
test_dir = PATH + '/test'
test_set = image_dataset_from_directory(directory=test_dir,
                                        image_size=TARGET_SIZE,
                                        batch_size=BATCH_SIZE,
                                        label_mode='categorical')
print('test_set: ', test_set)

for images, labels in training_set.take(1):
    print(f'\ntrain image shape: {images.shape[1:]},\ntrain label shape: {labels.shape[1:]}')
 
## Building a Transfer Learning feature extraction model using Keras Functional API
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

'''
NOTE:
The Sequential API is straight forward, it runs our layers in sequential order.
But the Functional API gives us more flexibility with our model. You can the doc
at https://tensorflow.org/guide/keras/functional
'''
 
# Create a base model
base_model = EfficientNetB0(include_top=False)

# Freeze the base model 
base_model.trainable = False
'''
NOTE: So the underlying pre-trained patterns aren't updated during training
'''

# Add input layer into the base model
inputs = Input(shape=TARGET_SIZE + (3,), name='input_layer')

# Normalize the inputs
# x = Rescaling(1./225)(inputs)
'''
NOTE:
A model like ResNet50V2 will need you to normalize your inputs, but you don't have
to for EfficientNet model. How do you know which one needed you to normalize your 
inputs? The answer is you won't know. You need to experince using that model itself
to know.
'''

# Pass the inputs into the base model
x = base_model(inputs)
print(f'\nShape after passing inputs through base model: {x.shape}')

# Average pool the outputs of the base model
x = GlobalAveragePooling2D(name='global_average_pooling_layer')(x)
print(f'Shape after GlobalAveragePooling2D: {x.shape}')
'''
NOTE: Aggregate all the most important information, and reduce the number of computation.
'''
 
# Add the output activation function into the base model
outputs = Dense(units=len(CLASS_NAMES), activation=softmax, name='output_layer')(x)

# Combine the inputs with the outputs into a model
efficientnet_model = Model(inputs, outputs, name='efficientnet_model')

# Compile the model
efficientnet_model.compile(optimizer=Adam(learning_rate=.01), 
                           loss=CategoricalCrossentropy(), 
                           metrics=['accuracy'])

# Model summary
print('\nModel Summary:')
efficientnet_model.summary()

# Fit the model
print(f'\nFitting {efficientnet_model.name}')
tensor_callback = myfunc.create_tensorboard_callback(dir_name='transfer_learning', 
                                                     experiment_name='efficientnet')
efficient_history = efficientnet_model.fit(training_set,
                                           epochs=5,
                                           steps_per_epoch=len(training_set),
                                           validation_data=test_set,
                                           validation_steps=len(training_set), 
                                           callbacks=tensor_callback)
'''
NOTE:
The reason why the validation_steps value is the length of training set is cecause 
the training set is smaller than the test set.
'''
 
# Evaluate the model
print(f'\nEvaluating {efficientnet_model.name}')
efficientnet_model.evaluate(test_set)
myfunc.plot_model_history(efficient_history)
 
# Getting a feature vector from a trained model
'''
NOTE: Demonstrating the Global Average Pooling 2D layer.
We have a tensor after out model goes through base_model of shape(None, None, None, 1280).
But then when it passes through GlobalAveragePooling2D, it turns into (None, 1280).
Let's use a similar shaped tensor of (1, 4, 4, 3) and then passes it into the 
GlobalAvearagePooling2D.
'''

# Define the input shape
input_shape = (1, 4, 4, 3)

# Create a random tensor
tf.random.set_seed(42)
input_tensor = tf.random.normal(input_shape)
print(f'\nRandom input tensor: {input_tensor.shape}')

# Pass the random tensor through a global averange pooling 2D layer
global_average_pooled_tensor = GlobalAveragePooling2D()(input_tensor)
print(f'\n2D global average pooled random tensor: {global_average_pooled_tensor.shape}\
      \n{global_average_pooled_tensor}')
 
# Replicate the GlobalAveragePool2D layer
print('Replicated 2D global average pooled tensor: \n', tf.reduce_mean(input_tensor, axis=[1, 2]))

'''
NOTE:
One of the reasons feature extraction transfer learning is named how it is because
what often happens is pretrained model outputs a feature vector (a long tensor of
numbers which represents the learned representation of the model on a particular
sample, in our case, this is the output of the tf.keras.layers.GlobalAveragePooling2D()
layer) which can then be used to extract patterns out of for our own specific problem.
'''
