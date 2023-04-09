## Comparing All The Models ##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import my_utils as utils
import model_0_naive_bayes as naive_bayes_model
import model_1_simple_dense as simple_dense_model
import model_2_lstm as lstm_model
import model_3_gru as gru_model
import model_4_bidirectional as bidirectional_model
import model_5_1d_conv as conv_1d_model
import model_6_nlp_transfer_learning as nlp_transfer_learning_model
import model_7_nlp_transfer_learning_10_percent_data as nlp_transfer_learning_10_percent_model

DF_PATH = '../data/nlp_getting_started'

## Building, fitting, and evaluating all the models
'''
NOTE:
You can run all these files individually.
'''
model_0, model_0_results = naive_bayes_model.run()
model_1_history, model_1_results = simple_dense_model.run()
model_2_history, model_2_results = lstm_model.run()
model_3_history, model_3_results = gru_model.run()
model_4_history, model_4_results = bidirectional_model.run()
model_5_history, model_5_results = conv_1d_model.run()
model_6_history, model_6_results = nlp_transfer_learning_model.run()
model_7_history, model_7_results = nlp_transfer_learning_10_percent_model.run()

model_1 = model_1_history.model
model_2 = model_2_history.model
model_3 = model_3_history.model
model_4 = model_4_history.model
model_5 = model_5_history.model
model_6 = model_6_history.model
model_7 = model_7_history.model

all_model_results = pd.DataFrame({model_0.name: model_0_results,
                                  model_1.name: model_1_results,
                                  model_2.name: model_2_results,
                                  model_3.name: model_3_results,
                                  model_4.name: model_4_results,
                                  model_5.name: model_5_results,
                                  model_6.name: model_6_results,
                                  model_7.name: model_7_results}).transpose()
print('\nAll model results:\n', all_model_results)

## Plot and compare all the model results
barwidth = 1/len(all_model_results.transpose().values[0])
plt.figure(figsize=(120, 80))
for idx in range(len(all_model_results.transpose().values)):
    row = all_model_results.transpose().values[idx]
    bar_position = [x + (idx * barwidth) for x in np.arange(len(row))]
    label = all_model_results.columns[idx]
    plt.bar(bar_position, row, width=barwidth, label=label)
plt.xlabel('Models', fontsize=240)
plt.ylabel('Result', fontsize=240)
plt.xticks([r + barwidth for r in range(len(all_model_results.values))],
           all_model_results.index,
           rotation=45,
           fontsize=140)
plt.yticks(fontsize=140)
plt.legend(loc=4, fontsize=140)
plt.title('All Model Results Diagram', fontsize=300)
plt.show()

## Plot f1-score of all models
accuracy_model_results = all_model_results['accuracy']
barwidth = 1/len(accuracy_model_results.values)
plt.figure(figsize=(120, 80))
for idx in range(len(accuracy_model_results.values)):
    row = accuracy_model_results.values[idx]
    plt.bar(idx, row)
plt.xlabel('Models', fontsize=240)
plt.ylabel('Accuracy', fontsize=240)
plt.xticks(np.arange(len(accuracy_model_results.values)),
           accuracy_model_results.index,
           rotation=45,
           fontsize=140)
plt.yticks(fontsize=140)
plt.title('All Model Accuracies Diagram', fontsize=300)
plt.show()

## The speed / score tradeoff

# Importing test dataset
test_set = pd.read_csv(f'{DF_PATH}/test.csv')

# Calculate all model time per preds
print('\nMaking predictions on the test dataset')
_, model_0_time_per_pred = utils.pred_timer(model_0, test_set.text.values)
_, model_1_time_per_pred = utils.pred_timer(model_1, test_set.text.values)
_, model_2_time_per_pred = utils.pred_timer(model_2, test_set.text.values)
_, model_3_time_per_pred = utils.pred_timer(model_3, test_set.text.values)
_, model_4_time_per_pred = utils.pred_timer(model_4, test_set.text.values)
_, model_5_time_per_pred = utils.pred_timer(model_5, test_set.text.values)
_, model_6_time_per_pred = utils.pred_timer(model_6, test_set.text.values)
_, model_7_time_per_pred = utils.pred_timer(model_7, test_set.text.values)

# Visualize them
plt.figure(figsize=(10, 8))
plt.scatter(model_0_time_per_pred, model_0_results['f1-score'], label=model_0.name)
plt.scatter(model_1_time_per_pred, model_1_results['f1-score'], label=model_1.name)
plt.scatter(model_2_time_per_pred, model_2_results['f1-score'], label=model_2.name)
plt.scatter(model_3_time_per_pred, model_3_results['f1-score'], label=model_3.name)
plt.scatter(model_4_time_per_pred, model_4_results['f1-score'], label=model_4.name)
plt.scatter(model_5_time_per_pred, model_5_results['f1-score'], label=model_5.name)
plt.scatter(model_6_time_per_pred, model_6_results['f1-score'], label=model_6.name)
plt.scatter(model_7_time_per_pred, model_7_results['f1-score'], label=model_7.name)
plt.title('F1-Score Vs Time per Prediction', fontsize=24)
plt.xlabel('Time per Prediction', fontsize=20)
plt.ylabel('F1-Score', fontsize=20)
plt.legend(bbox_to_anchor=(1., 1.), fontsize=16)
plt.show()
    