## Running a Seiries of Transfer Learning Experiments ##

'''
NOTE:
We've seen the incredible results Transfer Learning can get with only 10% of the
training data, but how does it go with 1% training data... These are a bunch of
experiments to find out:
1. model 1:     Use feature extraction transfer learning with 1% of the training 
                data with data augmentation.
2. model 2:     Use feature extraction transfer learning with 10% of the training
                data with data augmentation.
3. model 3:     Use fine-tuning transfer learning with 10% of the training data
                with data augmentation.
4. model 4:     Use fine-tuning transfer learning with 100% of the training data
                with data augmentation.
                
Throughout all experiments, the same test dataset will be used to evaluate the
models. This ensures consistency across evaluation metrics.
'''

import os
import tensorflow as tf
import my_functions as myfunc

print(f'\ntensorflow: {tf.__version__}')

OG_PATH = '../data/food-101/images'
PATH_100_PERCENT = '../data/food_101'
PATH_10_PERCENT = '../data/food_101_10_percent'
PATH_1_PERCENT = '../data/food_101_1_percent'

TARGET_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = ['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 
               'pizza', 'steak', 'hamburger', 'sushi', 'ramen', 'ice_cream']

INITIAL_EPOCHS = 5

## Preparing the data
import prepare_data as prep

# 100% training dataset
prep.prepare(OG_PATH, PATH_100_PERCENT,
             datasets=['train', 'test'],
             labels=CLASS_NAMES)

# Visualize the samples from 100% training dataset
file_dirs_100_percent = os.listdir(PATH_100_PERCENT + '/train')
filenames = []
for fdir in file_dirs_100_percent:
    for filename in os.listdir(f'{PATH_100_PERCENT}/train/{fdir}'):
        filenames.append(f'{fdir}/{filename}')
myfunc.plot_sample_images(filenames,
                          label_size=15,
                          dir_path=PATH_100_PERCENT + '/train',
                          read_from_dir=True)

# 10% training dataset
prep.prepare(OG_PATH, PATH_10_PERCENT, 
             datasets=['train', 'test'],
             labels=CLASS_NAMES,
             train_data_size=.1)

# Visualize the samples from 10% training dataset
file_dirs_10_percent = os.listdir(PATH_10_PERCENT + '/train')
filenames = []
for fdir in file_dirs_10_percent:
    for filename in os.listdir(f'{PATH_10_PERCENT}/train/{fdir}'):
        filenames.append(f'{fdir}/{filename}')
myfunc.plot_sample_images(filenames,
                          label_size=15,
                          dir_path=PATH_10_PERCENT + '/train',
                          read_from_dir=True)

# 1% training dataset
prep.prepare(OG_PATH, PATH_1_PERCENT,
             datasets=['train', 'test'],
             labels=CLASS_NAMES,
             train_data_size=.01)

# Visualize the samples from 1% training dataset
file_dirs_1_percent = os.listdir(PATH_1_PERCENT + '/train')
filenames = []
for fdir in file_dirs_1_percent:
    for filename in os.listdir(f'{PATH_1_PERCENT}/train/{fdir}'):
        filenames.append(f'{fdir}/{filename}')
myfunc.plot_sample_images(filenames,
                          label_size=15,
                          dir_path=PATH_1_PERCENT + '/train',
                          read_from_dir=True)

## Importing the datasets
from tensorflow.keras.preprocessing import image_dataset_from_directory

# 100% training set
training_set = image_dataset_from_directory(directory=PATH_100_PERCENT + '/train',
                                            image_size=TARGET_SIZE,
                                            batch_size=BATCH_SIZE,
                                            label_mode='categorical')

# 10% training set
training_set_10_percent = image_dataset_from_directory(directory=PATH_10_PERCENT + '/train',
                                                       image_size=TARGET_SIZE,
                                                       batch_size=BATCH_SIZE,
                                                       label_mode='categorical')

# 1% training set
training_set_1_percent = image_dataset_from_directory(directory=PATH_1_PERCENT + '/train',
                                                      image_size=TARGET_SIZE,
                                                      batch_size=BATCH_SIZE,
                                                      label_mode='categorical')

# test set
test_set = image_dataset_from_directory(directory=PATH_100_PERCENT + '/test',
                                        image_size=TARGET_SIZE,
                                        batch_size=BATCH_SIZE,
                                        label_mode='categorical')
'''
NOTE: It doesn't matter which test dataset path we choose because all the test 
datasets are same.
'''

## Adding data augmentation right into the model
'''
NOTE:
To add data augmentation right into the models, we can use layers inside:
*   tf.keras.layers.experimental.preprocessing()

We can see the benefits of doing this within the TensorFlow Data Augmentation
documentation: 
http://tensorflow.org/tutorials/images/data_augmentation#use_keras_preprocessing_layers
'''
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing

data_augmentation = Sequential(name='data_augmentation',
                               layers=[
                                   preprocessing.RandomFlip('horizontal'),
                                   preprocessing.RandomRotation(.2),
                                   preprocessing.RandomZoom(.2),
                                   preprocessing.RandomWidth(.2),
                                   preprocessing.RandomHeight(.2),
                                   # preprocessing.Rescale(1./225)
                                   ])

'''
NOTE:
Rescaling with data augmentation is useful for models like ResNet50V2. For models 
like EfficientNet, they already have a built-in rescaling.
    
Of the tops of out heads, after reading the docs, the benefits of using data augmentation
inside the model are:
*   Preprocessing of images (augmenting them) happens on the GPU (much faster)
    rather than the CPU.
*   Image data augmentation only happens during training, so we can still export
    our whole model and use it elsewhere.
'''

# Visualizing the augmented data
images, labels = list(training_set.take(1))[0]
augmented_images = data_augmentation(images, training=True)
myfunc.plot_augmented_data(augmented_data=augmented_images/255., 
                           original_data=images/255., 
                           with_keras_layers=True)

## Importing tensorflow libraries
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

INPUT_SHAPE = TARGET_SIZE + (3, )
TENSORBOARD_PATH = 'tmp/transfer_learning_experiments'
CHECKPOINTS_PATH = 'tmp/model_checkpoints'

## Model 1: Use feature extraction transfer learning with 1% of the augmented training data
model_1 = myfunc.create_feature_extraction_model(base_model=EfficientNetB0,
                                                 input_shape=INPUT_SHAPE,
                                                 data_augmentation=data_augmentation,
                                                 output_units=len(CLASS_NAMES),
                                                 output_activation=softmax,
                                                 model_name='model_1',
                                                 optimizer=Adam(),
                                                 loss=CategoricalCrossentropy())

# Fit the model
print('\nFitting model_1')
model_1_tensorboard_callback = myfunc.create_tensorboard_callback(dir_name=TENSORBOARD_PATH, 
                                                                  experiment_name='model_1')
model_1_checkpoint_callback = myfunc.create_model_checkpoint_callback(dir_name=CHECKPOINTS_PATH, 
                                                                      experiment_name='model_1')
model_1_history = model_1.fit(training_set_1_percent,
                              epochs=INITIAL_EPOCHS,
                              steps_per_epoch=len(training_set_1_percent),
                              validation_data=test_set,
                              validation_steps=len(training_set_1_percent),
                              callbacks=[model_1_tensorboard_callback, 
                                         model_1_checkpoint_callback])

# Evaluate the model
print('\nEvaluating model_1')
model_1_results = model_1.evaluate(test_set),
myfunc.plot_model_history(model_1_history)

# Load in checkpointed weights
model_1.load_weights(CHECKPOINTS_PATH + '/model_1')
loaded_weights_model_1_results = model_1.evaluate(test_set)
myfunc.compare_two_model_results(model_1_results, loaded_weights_model_1_results)

## Model 2: Use feature extraction transfer learning with 10% of the augmented training data
model_2 = myfunc.create_feature_extraction_model(base_model=EfficientNetB0,
                                                 input_shape=INPUT_SHAPE,
                                                 data_augmentation=data_augmentation,
                                                 output_units=len(CLASS_NAMES),
                                                 output_activation=softmax,
                                                 model_name='model_2',
                                                 optimizer=Adam(),
                                                 loss=CategoricalCrossentropy())

# Fitting the model
print('\nFitting model_2')
model_2_tensorboard_callback = myfunc.create_tensorboard_callback(dir_name=TENSORBOARD_PATH, 
                                                                  experiment_name='model_2')
model_2_checkpoint_callback = myfunc.create_model_checkpoint_callback(dir_name=CHECKPOINTS_PATH, 
                                                                      experiment_name='model_2')
model_2_history = model_2.fit(training_set_10_percent,
                              epochs=INITIAL_EPOCHS,
                              steps_per_epoch=len(training_set_10_percent),
                              validation_data=test_set,
                              validation_steps=len(training_set_10_percent),
                              callbacks=[model_2_tensorboard_callback,
                                         model_2_checkpoint_callback])

# Evaluating the model
print('\nEvaluating model_2')
model_2_results = model_2.evaluate(test_set)
myfunc.plot_model_history(model_2_history)

# Load in checkpointed weights
model_2.load_weights(CHECKPOINTS_PATH + '/model_2')
loaded_weights_model_2_results = model_2.evaluate(test_set)
myfunc.compare_two_model_results(model_2_results, loaded_weights_model_2_results)

## Model 3: Use fine-tuning transfer learning with 10% of the augmented training data
'''
NOTE:
Fine-tuning usually works best after training a feature extraction model for a few
epochs with a large amount of custom data.
'''

# Check if the layers inside the model_2 are trainable
print('\nmodel_2 layers and are those trainable:')
for layer in model_2.layers:
    print(layer, layer.trainable)

# Copy model_2 to new model (we will start using this new model for the fine-tuning experiment)
model_3 = model_2
model_3_base = model_3.layers[2]
print('\nbase_model total layers: ', len(model_3_base.layers))

# To begin fine-tuning, let's start by setting the last 10 layers of the 
# base_model.trainable = True
model_3_base.trainable = True
for layer in model_3_base.layers[:-10]:
    layer.trainable = False

print(f'\nTotal trainable layers in base_model of Model 3: {len([layer for layer in model_3_base.layers if layer.trainable])}')
model_3.layers[2] = model_3_base

# Recompile (we have to recompile the model everytime we make a change)
model_3.compile(optimizer=Adam(learning_rate=.0001),
                loss=CategoricalCrossentropy(),
                metrics=['accuracy'])
print("\nModel 3 total trainable variables: ", len(model_3.trainable_variables))
'''
NOTE:
When using fine-tuning it's best practice to lower your learning rate by some amount.
How much? This is a hyperparameter you can tune. But a good rule of thumb is at
least 10x (though different sources will claim other values). A good resource for
information on this is the ULMFiT paper: https://arxiv.org/abs/1801.06146
'''

# Fine tune for another 5 epochs
fine_tune_epochs = INITIAL_EPOCHS + 5

# Refit the model (same as model_2 except with more trainable layers)
print('\nFitting model_3')
model_3_tensorboard_callback = myfunc.create_tensorboard_callback(dir_name=TENSORBOARD_PATH, 
                                                                  experiment_name='model_3')
model_3_checkpoint_callback = myfunc.create_model_checkpoint_callback(dir_name=CHECKPOINTS_PATH, 
                                                                      experiment_name='model_3')
model_3_history = model_3.fit(training_set_10_percent,
                              epochs=fine_tune_epochs,
                              steps_per_epoch=len(training_set_10_percent),
                              validation_data=test_set,
                              validation_steps=len(training_set_10_percent),
                              initial_epoch=model_2_history.epoch[-1],
                              callbacks=[model_3_tensorboard_callback,
                                         model_3_checkpoint_callback])

# Evaluating the model
print('\nEvaluating model_3')
model_3.evaluate(test_set)
myfunc.plot_model_history(model_3_history)
myfunc.plot_2_model_histories(model_2_history, model_3_history)

'''
NOTE:
This is a little summary what we did with fine-tuning:
1.  Trained a feature extraction transfer learning model for 5 epoch on the 10% of
    the data with data augmentation (model_2) and saved the model's weights using
    ModelCheckpoint callback.
2.  Fine-tuned the same model on the same 10% of the data for a further 5 epochs 
    with the top 10 layers of the base_model being unfrozen (model_3)
3.  Saved the results and training logs each time
'''

## Model 4: Use fine-tuning transfer learning with 100% of the augmented training data

# Copy model_2 to a new model (we will name this new model model_4)
model_4 = model_2
model_4_base = model_4.layers[2]

# Un frozen the last 4 layers of the base model of model_4
model_4_base.trainable = True
for layer in model_4_base.layers[:-10]:
    layer.trainable = False

print(f'\nTotal trainable layers in base_model of Model 4: {len([layer for layer in model_4_base.layers if layer.trainable])}')
model_4.layers[2] = model_4_base

# Recompile the model
model_4.compile(optimizer=Adam(learning_rate=.0001),
                loss=CategoricalCrossentropy(),
                metrics=['accuracy'])
print("\nModel 4 total trainable variables: ", len(model_4.trainable_variables))

# Refit the model (same as model_2 except with more treainable layers)
print('\nFitting model_4')
model_4_tensorboard_callback = myfunc.create_tensorboard_callback(dir_name=TENSORBOARD_PATH, 
                                                                  experiment_name='model_4')
model_4_checkpoint_callback = myfunc.create_model_checkpoint_callback(dir_name=CHECKPOINTS_PATH, 
                                                                      experiment_name='model_4')
model_4_history = model_4.fit(training_set,
                              epochs=fine_tune_epochs,
                              steps_per_epoch=len(training_set),
                              validation_data=test_set,
                              validation_steps=len(test_set),
                              initial_epoch=model_2_history.epoch[-1],
                              callbacks=[model_4_tensorboard_callback,
                                         model_4_checkpoint_callback])

# Evaluating the model
print('\nEvaluating model_4')
model_4.evaluate(test_set)
myfunc.plot_model_history(model_4_history)
myfunc.plot_2_model_histories(model_3_history, model_4_history, 
                              model_1_name='model_3',
                              model_2_name='model_4')

## Upload TensorBoard dev records

# tensorboard dev upload \
#     --logdir ./tmp/transfer_learning_experiments/ \
#     --name "Transfer Learning Experiments" \
#     --description "Comparing 4 different experiment models with feature extraction and fine-tuning" \
#     --one_shot

## Checkout what TensorBoard experiments you have

# tensorboard dev list

## Delete an experiment

# tensorboard dev delete --experiment-id <experiment_id>
