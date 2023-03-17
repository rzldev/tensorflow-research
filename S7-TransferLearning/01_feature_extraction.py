## Transfer Learning - Feature Extraction ##

'''
NOTE:
Transfer learning is leveraging a working model's existing architecture and learned
patterns for our own problems. There are two main benefits of transfer learning.
1.  Can leverage existing Neural Networks architecture proven to work on similar
    problems.
2.  Can leverage a working Neural Networks architecture which has already learned
    patterns on similar data, then we can adapt those patterns to our data
'''

import os
import tensorflow as tf
import prepare_data as pd
import my_functions as myfunc

print(f'\ntensorflow: {tf.__version__}')

OG_PATH = '../data/food-101/images'
PATH = '../data/food_101_10_percent'

TARGET_SIZE = (64, 64)
BATCH_SIZE=32

CLASS_NAMES = ['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 
               'pizza', 'steak', 'hamburger', 'sushi', 'ramen', 'ice_cream']

## Preparing the data
pd.prepare(OG_PATH, PATH,
           datasets=['train', 'test'],
           labels=CLASS_NAMES,
           train_data_size=.1)

# Visualize the sample data
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training dataset
train_datagen = ImageDataGenerator(rescale=1./225)
training_set = train_datagen.flow_from_directory(PATH + '/train',
                                                 target_size=TARGET_SIZE,
                                                 batch_size=BATCH_SIZE,
                                                 class_mode='categorical')
print(f'training set size: {len(training_set)}')

# Test dataset
test_datagen = ImageDataGenerator(rescale=1./225)
test_set = test_datagen.flow_from_directory(PATH + '/test',
                                            target_size=TARGET_SIZE,
                                            batch_size=BATCH_SIZE,
                                            class_mode='categorical')
print(f'test set size: {len(test_set)}')

## Setting up TensorFlow callbacks
'''
NOTE:
Callbacks are extra functionality you can add to your models to be performed during 
or after training. Some of the most popular callbacks:
*   Tracking experiments with the TensorBoard callback.
*   Model checkpoint with the ModelCheckpoint callback.
*   Stopping a model from training (before it trains to long and overfit) with the
    EarlyStopping callback.
'''
import datetime
from tensorflow.keras.callbacks import TensorBoard

def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + '/' + experiment_name + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    print(f'\nSaving TensorBoard log files to: {log_dir}')
    return tensorboard_callback

## Creating models using TensorFlow Hub
'''
NOTE:
In the past we've used TensorFlow to create our own models, layer by layer from
scratch. Now we're going to do a similar process, except the majority of our model's
layer are going to come from TensorFlow Hub.

We can access pre-trained models on https://tfhub.dev/
Browsing the TensorFlow Hub pages and sorting for image classification, we found
the following feature vector model link:
https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1/
'''

import tensorflow_hub as hub
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import sigmoid, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy

resnet_url = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4'
efficientnet_url = 'https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1'

## Create TensorBoard callback (functionized because we need to create a new one for each model)
def create_model(model_url, num_classes=2, output_activation=sigmoid, model_name='simple_model',
                 optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy']):
    '''
    Takes a TensorFlow Hub URL and creates Keras Sequential model with it.
    
    Args:
        model_url (str): A TensorFlow Hub feature extraction URL.
        num_classes (int): Number of output neurons in the output layer, should
            be equal to number of target classes, default 2.
        output_activation (str): Activation function that will be used for the
            output layer, default sigmoid.
        model_name (str): A name for the model, default "simple_model".
            
    Returns:
        An unfitted Keras Sequential model with model URL as feature extractor
        layer and Dense output layer with output activation function and num_classes 
        output neurons.
    '''
    
    # Download the pretrained model and save it as Keras layer
    feature_extractor_layer = hub.KerasLayer(model_url,
                                             trainable=False,
                                             name='feature_extractor_layer',
                                             input_shape=TARGET_SIZE + (3,))
    
    # Create the model
    model = Sequential(name=model_name)
    
    # The layers to the model
    model.add(feature_extractor_layer)
    model.add(Dense(units=num_classes, activation=output_activation, name='output_layer'))
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model

## Build Resnet model
# Create the model
resnet_model = create_model(resnet_url, 
                            num_classes=len(CLASS_NAMES), 
                            output_activation=softmax,
                            model_name='resnet_model',
                            optimizer=Adam(), 
                            loss=CategoricalCrossentropy(), 
                            metrics=['accuracy'])

# Model summary
print('\nModel Summary:')
resnet_model.summary()

# Fit the model into the training set
resnet_history = resnet_model.fit(training_set,
                                  epochs=5,
                                  steps_per_epoch=len(training_set),
                                  validation_data=test_set,
                                  validation_steps=len(test_set),
                                  callbacks=[create_tensorboard_callback(dir_name='tensorflow_hub', 
                                                                         experiment_name='resnet50v2')])

# Evaluate the model
print(f'\nEvaluating the {resnet_model.name}')
resnet_model.evaluate(test_set)
myfunc.plot_model_history(resnet_history)

## Build Efficientnet model
# Create the model
efficientnet_model = create_model(efficientnet_url,
                                  num_classes=len(CLASS_NAMES),
                                  output_activation=softmax,
                                  model_name='efficientnet_model',
                                  optimizer=Adam(), 
                                  loss=CategoricalCrossentropy(), 
                                  metrics=['accuracy'])

# Model summary
print('\nModel Summary:')
efficientnet_model.summary()

# Fit the model into the training set
efficientnet_history = efficientnet_model.fit(training_set,
                                              epochs=5,
                                              steps_per_epoch=len(training_set),
                                              validation_data=test_set,
                                              validation_steps=len(test_set),
                                              callbacks=[create_tensorboard_callback(dir_name='tensorflow_hub', 
                                                                                     experiment_name='efficientnetb0')])

# Evaluating the model
print(f'\nEvaluating {efficientnet_model.name}')
efficientnet_model.evaluate(test_set)
myfunc.plot_model_history(efficientnet_history)

## Different types of Transfer Learning
'''
NOTE:
These are the different types of Transfer Learning.
*   "As is" - using an existing model with no changes whatsoever (e.g. using ImageNet 
              model on 1000 ImageNet classes, None of your own)
*   "Feature extraction" - use the prelearned patterns of an existing model (e.g.
                           EfficientNetB0 trained on ImageNet) and adjust the output
                           layer for your own problem (e.g. 1000 classes -> 10 classes
                           of food)
*   "Fine-tuning" - use a prelearned patterns of an existing model and "fine-tune"
                    many of all the underlying layers (including new output layers)
'''

## Comparing our models results using TensorBoard
'''
NOTE:
When you upload things to TensorBoard.dev, your experiments are public. So, if you're
running private experiments (things you don't want others to see) do not upload 
them to TensorBoard.dev.
'''

## Upload TensorBoard dev records

# tensorboard dev upload \
#     --logdir ./tensorflow_hub/ \
#     --name "EfficientNetB0 vs ResNet50V2" \
#     --description "Comparing two different TF Hub Feature Extraction model architectures using 10% of the training data" \
#     --one_shot

## Checkout what TensorBoard experiments you have

# tensorboard dev list

## Delete an experiment

# tensorboard dev delete --experiment-id <experiment_id>
