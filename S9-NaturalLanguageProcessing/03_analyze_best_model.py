## Analyzing the Best Model ##

import numpy as np
import pandas as pd
import tensorflow as tf
import random as rand
import my_utils as utils
import model_6_nlp_transfer_learning as nlp_transfer_learning_model

print(f'\ntensorflow: {tf.__version__}')

DF_PATH = '../data/nlp_getting_started'

## Importing the dataset
training_set = pd.read_csv(f'{DF_PATH}/train.csv')
test_set = pd.read_csv(f'{DF_PATH}/test.csv')
print('\nTraining set:\n', training_set.head())
print('\nTest set:\n', test_set.head())

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split

training_texts, validation_texts, training_labels, validation_labels = train_test_split(training_set.text.values,
                                                                                        training_set.target.values,
                                                                                        test_size=.2,
                                                                                        random_state=42)

## Call the best model (Model 6: Transfer Learning for NLP)
model_6_history, _ = nlp_transfer_learning_model.run()
model_6 = model_6_history.model

## Evaluate the model
print(f'\nEvaluating {model_6.name}')
model_6.evaluate(validation_texts, validation_labels)
utils.plot_model_history(model_6_history)

## Making predictions with validation dataset
print(f'\nMaking predictions using {model_6.name}')
model_6_pred_probs = model_6.predict(validation_texts)
model_6_preds = tf.squeeze(tf.round(model_6_pred_probs))
model_6_results = utils.evaluate_model_results(validation_labels, model_6_preds)
print(f'\nModel 6 (Universal Sentence Encoder) results: \n{model_6_results}')
utils.plot_confusion_matrix(validation_labels, model_6_preds)

# Create DataFrame with validation texts, validation labels, and best performing
# model prediction labels + probabilities
validation_df = pd.DataFrame({'text': validation_texts,
                              'label': validation_labels,
                              'preds': model_6_preds,
                              'probs': tf.squeeze(model_6_pred_probs)})

# Most wrong predictions
print()
most_wrong_df = validation_df[validation_df['label'] != validation_df['preds']].sort_values('probs', ascending=False)
print(most_wrong_df.head())
print(most_wrong_df.tail())

# Visualize worst predictions on validation dataset
random_validation_data = validation_df.sample(n=10)
for row in random_validation_data.values:
    text, label, pred, prob = row
    print(f'\nTarget: {label}, pred: {pred}, prob: {prob}')
    print(f'Text: \n{text}\n')
    print('\n----------\n')
    
## Making predictions with test set
print(f'\nMaking predictions using {model_6.name} on test dataset')
model_6_test_pred_probs = model_6.predict(test_set.text.values)
model_6_test_preds = tf.squeeze(tf.round(model_6_test_pred_probs))

# Create DataFrame with test text, predicted classnames, and probabilities
test_df = pd.DataFrame({'text': test_set.text.values,
                        'preds': model_6_test_preds,
                        'probs': tf.squeeze(model_6_test_pred_probs)})
test_df['class_name'] = np.where(test_df['preds'] == 1, 'Diaster', 'Not Diaster')

# Visualize predictions on test dataset
random_test_data = test_df.sample(n=10)
for row in random_test_data.values:
    text, pred, prob, class_name = row
    print(f'\nLabel: {class_name}, pred: {pred}, prob: {prob:5f}')
    print(f'Text: \n{text}\n')
    print('\n----------\n')
































