## Food Vision Project ##

import tensorflow as tf
import utils

print(f'\ntensorflow: {tf.__version__}')

IMG_SIZE = (224, 224)
INPUT_SHAPE = IMG_SIZE + (3,)

TENSORBOARD_PATH = './tmp/tensorboards'
CHECKPOINT_PATH = './tmp/checkpoints'

INITIAL_EPOCHS = 3

## Use TensorFlow Datasets to download the data
import tensorflow_datasets as tfds

'''
NOTE:
If you want to get the overview of TensorFlow Datasets (TFDS), read the guide:
https://tensorflow.org/datasets/overview
'''

# List all availabel datasets and check if the dataset we want exists
dataset_list = tfds.list_builders()
print(f'\nTensorFlow Datasets samples: \n{dataset_list[:10]}')
print(f'\nfood101 datasets exists: {"food101" in dataset_list}')

# Load in the dataset
(train_data, test_data), ds_info = tfds.load(name='food101',
                                             split=['train', 'validation'],
                                             shuffle_files=True,
                                             as_supervised=True, # Data gets returned in tuple format (data, label)
                                             with_info=True)

# Exploring the dataset
print('\nDataset features:\n', ds_info.features)
food101_class_names = ds_info.features['label'].names
print('\nDataset class names:\n', food101_class_names[:10])

train_sample = train_data.take(1)
print('\nHow the train sample data looks like:\n', train_sample)
for image, label in train_sample:
    print(f'''\nData info:
          Image shape: {image.shape} 
          Image datatype: {image.dtype}
          Target class from Food101 (Tensor form): {label}
          Class name (str form): {food101_class_names[label.numpy()]}
          ''')
    print(f'The min and max value of the image Tensor: {tf.reduce_min(image)}, {tf.reduce_max(image)}')

    # Plot the sample image
    utils.plot_sample_images(image, food101_class_names[label.numpy()], title_size=20)
    
## Prerocess the data
'''
NOTE:
Neural Networks perform best when the data is in a certain way (e.g. batched, normalized, 
etc). However, not at data (including data from TensorFlow datasets) comes like this. 
So in order to get it ready for neural netwoks, you'll often have to write preprocessing
functions and map it to your data.

What we know about our data:
*   Using uint8 dtype
*   Comprised of all different size tensors (different size images)
*   Not scaled (the pixel values are between 0 & 255)

What we know models like:
*   Data in float32 dtype (or for mixed precision float16 and float32)
*   For batches, TensorFlow likes all the tensors within a batch to be of the same
    size
*   Scaled (values between 0 & 1) also called normalized tensors generally to perform
    better
    
With those points above, we've got a few things we can tackle with preprocessing
function. Since we are going to use EfficientNetBX pretrained model from tf.keras.applications
we don't need to rescale our data (these architectures have rescaling built-in)

This means the function needs to:
1.  Reshape our images to all the same size
2.  Convert the dtype of our images tensors from uint8 to float32
'''

for img, label in train_sample:
    preprocess_img = utils.preprocess_image(image, label)[0]
    print(f'\nImage before preprocessing: \n{img[:2]} \nShape: {img.shape} \nDatatype: {img.dtype}')
    print(f'\nImage after preprocessing: \n{preprocess_img[:2]} \nShape: {preprocess_img.shape} \nDatatype: {preprocess_img.dtype}')
    
## Batch & prepare datasets
from tensorflow.data import AUTOTUNE
'''
We're now going to make our data input pipeline runs really fast. For more resources
on this, you can going through this following guide: https://tensorflow.org/guide/data_performance
'''
    
# Preprocess training data
train_data = train_data.map(map_func=utils.preprocess_image, num_parallel_calls=AUTOTUNE)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=AUTOTUNE)

# Preprocess test data
test_data = test_data.map(map_func=utils.preprocess_image, num_parallel_calls=AUTOTUNE).batch(32).prefetch(buffer_size=AUTOTUNE)

print(f'\nTrain data: {train_data}')
print(f'\nTest data: {test_data}')

## Setup mixed precision training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

'''
NOTE:
For a deeper understanding of mixed precision training, check out the TensorFlow
guide for mixed precision: https://tensorflow.org/guide/mixed_precision

Mixed precision utilizes a combination of float32 and float16 data types to speed
up model performance. We won't use mixed precision because we don't have the required
GPU to run it otherwise our training performance will run slower. We need 7.0+ 
GPU score to run mixed precision.
'''

## Build feature extraction model
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Activation
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Create the base model and freeze underlying layers
base_model = EfficientNetB0(include_top=False)
base_model.trainable = False

# Create a functional model
inputs = Input(shape=INPUT_SHAPE, name='input_layer')
'''
NOTE: 
EfficientNetBX models have rescaling built-in, but if your model doesn't you can
have a layer like down below
'''
# x = preprocessing.Rescaling(1/255.)(x)
x = base_model(inputs, training=False) # Make sure layers which should be in inference mode only stay like that
x = GlobalAveragePooling2D()(x)
x = Dense(units=len(food101_class_names))(x)
outputs = Activation(softmax, dtype=tf.float32, name='softmax_float32')(x)
model = Model(inputs, outputs, name='food101_model')

# Compile the model
model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(),
              metrics='accuracy')

# Model summary
print('\nModel Summary:')
model.summary()

# Mixed precision policy
print(f'\nMixed precision policy: {mixed_precision.global_policy}')

# Checking layer dtype policies (are we using mixed precision?)
print('\nModel layers:')
for layer in model.layers:
  print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

# Check the dtype_policy attributes of layers inside the base_model
print('\nThe last 10 of base model layers:')
for layer in model.layers[1].layers[-10:]:
  print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

'''
NOTE:
layer.name: the human readable name of a particular layer
layer.trainable: is the layer trainable or not? (if False, the weights are frozen)
layer.dtype: the datatype of a layer stores its variables in
layer.dtype_policy: the datatype policy a layer computes on its variable with
'''

# Fit the model into the training data
print('\nFitting the model')
feature_extraction_tensorboard = utils.create_tensorboard_callback(dir_name=TENSORBOARD_PATH, 
                                                                   experiment_name=model.name)
feature_extraction_checkpoint = utils.create_model_checkpoint_callback(dir_name=CHECKPOINT_PATH, 
                                                                       experiment_name=model.name,
                                                                       monitor='val_accuracy')
feature_extraction_model_history = model.fit(train_data,
                                             epochs=INITIAL_EPOCHS,
                                             steps_per_epoch=len(train_data),
                                             validation_data=test_data,
                                             validation_steps=(.2 * len(test_data)),
                                             callbacks=[feature_extraction_tensorboard,
                                                        feature_extraction_checkpoint])

# Evaluate the model
print('\nEvaluating the model')
model.evaluate(test_data)
utils.plot_model_history(feature_extraction_model_history)

## Fine-tuning

# Check if the layers inside the model are trainable
print('\nModel trainable layers:')
for layer in model.layers:
  print(layer.name, layer.trainable)

# Setting all of the base_model layers to be trainable
model.trainable = True

trainable_layers = [layer for layer in model.layers[1].layers if layer.trainable]
print(f'\nTotal layers inside the base model: {len(model.layers[1].layers)}')
print(f'Total trainable layers inside the base model: {len(trainable_layers)}')

# Recompile the fine-tuned model
model.compile(optimizer=Adam(learning_rate=.0001),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Fine-tuned model summary
print('\nModel Summary:')
model.summary()

# Refit the model
print('\nFitting the model')
fine_tuned_tensorboard = utils.create_tensorboard_callback(dir_name=TENSORBOARD_PATH,
                                                           experiment_name=f'fine_tuned_{model.name}')
fine_tuned_checkpoint = utils.create_model_checkpoint_callback(dir_name=CHECKPOINT_PATH,
                                                               experiment_name=f'fine_tuned_{model.name}')
fine_tuned_early_stopping = utils.create_early_stopping_callback(patience=2)
fine_tuned_reduce_lr = utils.create_reduce_lr_callback(patience=0)
fine_tuned_model_history = model.fit(train_data,
                                     epochs=25,
                                     steps_per_epoch=len(train_data),
                                     validation_data=test_data,
                                     validation_steps=(.2*len(test_data)),
                                     initial_epoch=feature_extraction_model_history.epoch[-1],
                                     callbacks=[fine_tuned_tensorboard, 
                                                fine_tuned_checkpoint,
                                                fine_tuned_early_stopping,
                                                fine_tuned_reduce_lr])

# Evaluate the model
print('\nEvaluating the model')
model.evaluate(test_data)
utils.plot_model_history(fine_tuned_model_history)
utils.plot_2_model_histories(fine_tuned_model_history, feature_extraction_model_history,
                             model_1_name=f'fine-tuned {model.name}')

## Making predictions
print(f'\nMaking predictions with {model.name}')
pred_probs = model.predict(test_data, verbose=1)
pred_classes = pred_probs.argmax(axis=1)

print(f'\nNumber of prediction probabilities for sample 0: {len(pred_probs[0])}')
print(f'What prediction probability sample 0 looks like: \n{pred_probs[0]}')
print(f'The class with highest predicted probability by the model for sample 0: \
{pred_classes[0]} ({food101_class_names[pred_classes[0]]})')

## Compare the model's predictions with the original dataset labels
from sklearn.metrics import accuracy_score

y_labels = []
for _, labels in test_data.unbatch():
  y_labels.append(labels.numpy())

print(f'\nAccuracy score: {accuracy_score(y_labels, pred_classes)}')

## Making confusion matrix
utils.plot_confusion_matrix(y_labels, pred_classes, food101_class_names,
                            title=f'{model.name} Confusion Matrix',
                            figsize=(150, 150),
                            title_size=120,
                            label_size=80)
