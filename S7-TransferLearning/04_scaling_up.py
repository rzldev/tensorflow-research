## Transfer Learning - Scaling Up ##
'''
NOTE:
We've seen the power of transfer learning feature extraction and fine-tuning, now
it's time to scale up to all of the classes in Food101 (101 total classes). The
goal is to beat the original Food101 paper with the 10% of training (leveraging the 
power of deep learning). The baseline to beat is 50.76% accuracy across 101 classes.
https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf
'''

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import my_functions as myfunc
import prepare_data as prep
import matplotlib.pyplot as plt
import matplotlib.image as mimg

print(f'\ntensorflow: {tf.__version__}')

OG_PATH = '../data/food-101/images'
PATH_10_PERCENT = '../data/food_101_10_percent_all_classes'
CUSTOM_PATH = '../data/custom_food_data'

IMAGE_SIZE = (224, 224)
TARGET_SIZE = IMAGE_SIZE + (3,)
BATCH_SIZE = 32
INITIAL_EPOCHS = 5

TENSORBOARD_PATH = './tmp/transfer_learning/scaling_up/tensorboard'
CHECKPOINT_PATH = './tmp/transfer_learning/scaling_up/checkpoints'
SAVE_MODEL_PATH = './tmp/saved_models'

## Preparing the data
food_class_names = list(prep.get_labels(OG_PATH, filename='train.json'))

prep.prepare(OG_PATH, PATH_10_PERCENT,
             datasets=['train', 'test'],
             labels=food_class_names,
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

## Importing the datasets
from tensorflow.keras.preprocessing import image_dataset_from_directory

# 10% training set
training_set_10_percent = image_dataset_from_directory(PATH_10_PERCENT + '/train',
                                                       image_size=IMAGE_SIZE,
                                                       batch_size=32,
                                                       label_mode='categorical')

# test set
test_set = image_dataset_from_directory(PATH_10_PERCENT + '/test',
                                        image_size=IMAGE_SIZE,
                                        batch_size=BATCH_SIZE,
                                        label_mode='categorical',
                                        shuffle=False)

## Prepare data augmentation layer to add to the base model
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing

# Creating the data augmentation layer
data_augmentation = Sequential(name='data_augmentation', layers=[
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomZoom(.2),
    preprocessing.RandomWidth(.2),
    preprocessing.RandomHeight(.2),
    preprocessing.RandomRotation(.2),
    # preprocessing.Rescale(1./225)
    ])

# Visualize the sample augmented data
images, labels = list(training_set_10_percent.take(1))[0]
augmented_images = data_augmentation(images, training=True)
myfunc.plot_augmented_data(augmented_data=augmented_images/255., 
                           original_data=images/255.,
                           with_keras_layers=True)

## Build and train transfer learning model with 10% of 101 food classes
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# Setup the base model and freeze its layers (this will extract features)
base_model = EfficientNetB0(include_top=False)
base_model.trainable = False

# Setup model architecture with trainbale top layers
inputs = Input(shape=TARGET_SIZE, name='input_layer')
x = data_augmentation(inputs) # augment images (only happens during training phase)
x = base_model(x, training=False) # Put the base model in inference mode so the weights which needed will stay frozen
x = GlobalAveragePooling2D(name='global_average_pooling_layer')(x)
outputs = Dense(units=len(food_class_names), activation=softmax, name='output_layer')(x)
model = Model(inputs, outputs, name='food_101_classification_model')

# Compile the model
model.compile(optimizer=Adam(learning_rate=.001), 
              loss=CategoricalCrossentropy(), 
              metrics=['accuracy'])

# Model Summary
print('\nModel Summary:')
model.summary()

# Fit the model into the training set
print(f'\nFitting {model.name}')
model_tensorboard_callback = myfunc.create_tensorboard_callback(dir_name=TENSORBOARD_PATH, 
                                                                experiment_name='model')
model_checkpoint_callback = myfunc.create_model_checkpoint_callback(dir_name=CHECKPOINT_PATH, 
                                                                    experiment_name='model')
model_history = model.fit(training_set_10_percent,
                          epochs=INITIAL_EPOCHS,
                          steps_per_epoch=len(training_set_10_percent),
                          validation_data=test_set,
                          validation_steps=len(training_set_10_percent),
                           callbacks=[model_tensorboard_callback, model_checkpoint_callback])

# Evaluate the model
print(f'Evaluating {model.name}')
model.evaluate(test_set)
myfunc.plot_model_history(model_history)

'''
NOTE:
If the val-accuracy plot is lower that then the training_accuracy, there is probability
that model is overfitting. Ideally, the two curves should be very similar to each
other. We can try to improve the model performance with fine-tuning.
'''

## Fine-tuning
from tensorflow.keras.models import load_model

# Check if the layers inside the model are trainable
for layer in model.layers:
    print(layer.name, layer.trainable)
    
# Setting the last 5 layers of the base_model to be trainable
model.layers[2].trainable = True
for layer in model.layers[2].layers[:-10]:
    layer.trainable = False
    
print(f'\nTotal layers inside the base_model: {len(model.layers[2].layers)}')
print('\nTrainable layers inside the base_model:')
for layer_idx, layer in enumerate(model.layers[2].layers):
    if layer.trainable == True:
        print((layer_idx + 1), layer.name, layer.trainable)

# Recompile the model (we have to recompile the model everytime me make some changes)
model.compile(optimizer=Adam(learning_rate=.0001), # Learning rate lowered by 10x
              loss=CategoricalCrossentropy(),
              metrics=['accuracy'])

# Refit our fine-tuned model
print(f'\nFitting fine-tuned {model.name}')
fine_tune_epochs = INITIAL_EPOCHS + 5
fine_tuning_tensorboard_callback = myfunc.create_tensorboard_callback(dir_name=TENSORBOARD_PATH, 
                                                                      experiment_name='fine_tuning_model')
fine_tuning_checkpoint_callback = myfunc.create_model_checkpoint_callback(dir_name=CHECKPOINT_PATH, 
                                                                          experiment_name='fine_tuning_model')
fine_tuning_history = model.fit(training_set_10_percent,
                                epochs=fine_tune_epochs,
                                steps_per_epoch=len(training_set_10_percent),
                                validation_data=test_set,
                                validation_steps=len(training_set_10_percent),
                                initial_epoch=model_history.epoch[-1],
                                callbacks=[fine_tuning_tensorboard_callback,
                                            fine_tuning_checkpoint_callback])

# Evaluate the fine-tuned model
print(f'\nEvaluating fine-tuned {model.name}')
fine_tuning_evaluation = model.evaluate(test_set)
myfunc.plot_model_history(fine_tuning_history)
myfunc.plot_2_model_histories(fine_tuning_history, model_history, 
                              model_1_name=('fine-tuned ' + model.name))

# Save and load our saved model
model.save(SAVE_MODEL_PATH + model.name)
loaded_model = load_model(SAVE_MODEL_PATH + model.name)

# Evaluate the loaded model
print('\nEvaluating loaded model')
loaded_model_evaluation = loaded_model.evaluate(test_set)
myfunc.compare_two_model_results(fine_tuning_evaluation, loaded_model_evaluation)

## Making predictions
print(f'\nMaking predicting with {model.name}')
pred_probs = model.predict(test_set, verbose=1)

print(f'Number of prediction probabities for sample 0: {len(pred_probs[0])}')
print(f'What prediction probability sample 0 looks like: \n{pred_probs[0]}')
print(f'The class with the highest predicted probability by the model for sample 0: {pred_probs[0].argmax()} \
      ({food_class_names[pred_probs[0].argmax()]})')

# Get the pred classes of each label
pred_classes = pred_probs.argmax(axis=1)

## Compare the model's predictions with the original test dataset labels
from sklearn.metrics import accuracy_score

y_labels = []
for _, labels in test_set.unbatch():
    y_labels.append(labels.numpy().argmax())
print(len(y_labels), len(pred_classes))

pred_accuracy = accuracy_score(y_labels, pred_classes)
print("\nPrediction accuracy: ", pred_accuracy)

## Making confusion matrix plot
myfunc.plot_confusion_matrix(y_labels, pred_classes, food_class_names,
                             title=f'{model.name} Confusion Matrix',
                             figsize=(150, 150),
                             title_size=120,
                             label_size=80,
                             savefig=True,
                             save_path='./tmp/')

## Classification report
'''
NOTE:
Scikit-learn has a helpful function for aquiring many different classification
metrics per class (e.g. precision, recall and F1) called classification report,
let's try it out.
'''

from sklearn.metrics import classification_report
print(f'Classification report: \n{classification_report(y_labels, pred_classes)}')

# Plot the f1-score from classification report
model_report_dict = classification_report(y_labels, pred_classes, output_dict=True)
classification_f1_dict = {}
for k, v in model_report_dict.items():
    if k == 'accuracy':
        break
    else:
        classification_f1_dict[food_class_names[int(k)]] = v['f1-score']

f1_df = pd.DataFrame({'class_names': classification_f1_dict.keys(),
                      'f1_scores': classification_f1_dict.values()}).sort_values('f1_scores', ascending=False)
myfunc.plot_f1_scores(f1_df, title=f'F1-Score of {model.name}')

## Visualizing predictions on test images
'''
NOTE:
To visualize our model's predictions on our own images, we'll need a function
to load and preprocess the images. Specially, it will need to:
*   Read in a target image file path using tf.io.read_file()
*   Turn the image into tensor using tf.io.decode_image()
*   Resize the image tensor to be the same size as the images in our model using
    tf.image.resize()
*   Scale the image to get all of the pixel values from 0 to 1 (if necessary)
*   Make prediction on the loaded sample image from the test dataset
*   Plot the image along with the model's prediction
'''
test_images = prep.get_labels(OG_PATH, filename='test.json')
test_images = np.array(list(test_images.values())).flatten()
image_format = np.vectorize(lambda x: x + '.jpg')
test_images = image_format(test_images)

for _ in range(10):
    random_image_path = f'{PATH_10_PERCENT}/test/{np.random.choice(test_images)}'
    print(random_image_path)
    myfunc.predict_and_visualize(model, 
                                 labels=food_class_names, 
                                 img_path=random_image_path,
                                 title_size=16)

## Finding the most wrong predictions
'''
NOTE:
To find out where our model's worst, let's write some code to find out the following:
1.  Get all of the image file paths in test dataset.
2.  Create a pandas DataFrame of the image file paths, ground truth labels, predicted
    classes (from our model), max prediction probs.
3.  Use our DataFrame to find all the wrong predictions (where the ground truth 
    label doesn't match the prediction).
4.  Sort the DataFrame based on the wrong predictions (from the highest prediction
    probability to the lowest).
5.  Visualize the images with the highest prediction probabilities but have the
    wrong prediction.
'''

# Get all the image file paths in test dataset
filepaths = []
for filepath in test_set.list_files(f'{PATH_10_PERCENT}/test/*/*.jpg', shuffle=False):
    filepaths.append(filepath.numpy())
print(filepaths[:5])

# Create DataFrame of different parameters for each of the test images
pred_df = pd.DataFrame({'img_path': filepaths,
                        'y_true': y_labels,
                        'y_pred': pred_classes,
                        'pred_conf': pred_probs.max(axis=1),
                        'y_true_classname': [food_class_names[y] for y in y_labels],
                        'y_pred_classname': [food_class_names[y] for y in pred_classes]})
print(pred_df.head())

# Find out in our DataFrame which prediction are wrong
pred_df['pred_correct'] = pred_df['y_true'] == pred_df['y_pred']
print(pred_df.head())

# Sort the DataFrame to have the worst prediction at the top
worst_100 = pred_df[pred_df['pred_correct'] == False].sort_values('pred_conf', 
                                                                  ascending=False)[:100]
print(worst_100.head())

# Visualize the test data samples which has the worst prediction but highest pred probability
myfunc.visualize_image_prediction(worst_100, images_to_view=12)

# Make predictions and plot custom food images
custom_food_images = [f'{CUSTOM_PATH}/{path}' for path in os.listdir(CUSTOM_PATH)]
for img_path in custom_food_images:
    myfunc.predict_and_visualize(model, labels=food_class_names, img_path=img_path)
