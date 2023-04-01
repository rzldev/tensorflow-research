## Helpful Function ##

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import random
import datetime
import itertools

from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

def plot_sample_images(image, label, figsize=(6, 6), title_size=12):
    '''
    Plotting image samples.
    '''
    print('\nCreating images sample plot')
    plt.figure(figsize=figsize)
    plt.imshow(image),
    plt.title(label, fontsize=title_size)
    plt.axis(False)
    plt.show()
    
def preprocess_image(image, label, target_shape=(224, 224), rescale=False):
    '''
    Converts image datatype from `uint8` to `float32`, rescale the image if necessary 
    and reshapes image to [img_shape, img_shape, colour_channels]
    '''
    image = tf.image.resize(image, size=target_shape) # Reshape target image
    if rescale:
        image = image/255.
    return tf.cast(image, tf.float32), label # Return (float32_image, label) tuple

def create_tensorboard_callback(dir_name, experiment_name):
    '''
    Create and return a tensorboard callback which can be used while fitting the
    model.
    '''
    log_path = dir_name + '/' + experiment_name + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = TensorBoard(log_dir=log_path)
    print(f'\nSaving TensorBoard log files to: {log_path}')
    return tensorboard_callback

def create_model_checkpoint_callback(dir_name, experiment_name, monitor='val_loss', 
                                     save_weights_only=True, save_best_only=False):
    '''
    Create and return a model checkpoint callback which can be used while fitting 
    the model.
    '''
    checkpoint_path = dir_name + '/' + experiment_name
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                       monitor=monitor,
                                       save_weights_only=save_weights_only,
                                       save_best_only=save_best_only,
                                       save_freq='epoch', # Save for every epoch
                                       verbose=1)
    return model_checkpoint

def create_early_stopping_callback(monitor='val_loss', patience=3):
    '''
    Create and return an early stopping callback which will stop the training when 
    a monitored metric has stopped improving.
    '''
    early_stopping = EarlyStopping(monitor=monitor, 
                                   patience=patience)
    return early_stopping

def plot_model_history(history):
    '''
    Returns a separate loss and accuracy curves for validation metrics
    '''
    model_name = history.model.name
    print(f'\nPlotting {model_name} history')
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(history.history['loss']))
    
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title(f'{model_name} Loss Curve Plot')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title(f'{model_name} Accuracy Curve Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_2_model_histories(model_1_history, model_2_history, model_1_name=None, 
                           model_2_name=None):
    '''
    Returns a separate loss and accuracy curves for validation metrics
    '''
    model_1_name = (model_1_name if model_1_name != None else model_1_history.model.name)
    model_2_name = (model_2_name if model_2_name != None else model_2_history.model.name)
    print(f'\nPlotting and comparing {model_1_name} and {model_2_name} histories')
    
    history_1 = model_1_history.history
    history_2 = model_2_history.history
    epochs_1 = range(len(history_1['loss']))
    epochs_2 = range(len(history_2['loss']))
    
    plt.plot(epochs_1, history_1['loss'], c='red', label=f'{model_1_name} training_loss')
    plt.plot(epochs_1, history_1['val_loss'], c='salmon', label=f'{model_1_name} val_loss')
    plt.plot(epochs_2, history_2['loss'], c='gold', label=f'{model_2_name} training_loss')
    plt.plot(epochs_2, history_2['val_loss'], c='goldenrod', label=f'{model_2_name} val_loss')
    plt.title(f'{model_1_name} and {model_2_name} Loss Curve Plots')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
    plt.plot(epochs_1, history_1['accuracy'], c='green', label=f'{model_1_name} training_accuracy')
    plt.plot(epochs_1, history_1['val_accuracy'], c='lime', label=f'{model_1_name} val_accuracy')
    plt.plot(epochs_2, history_2['accuracy'], c='darkorange', label=f'{model_2_name} training_accuracy')
    plt.plot(epochs_2, history_2['val_accuracy'], c='sandybrown', label=f'{model_2_name} val_accuracy')
    plt.title(f'{model_1_name} and {model_2_name} Accuracy Curve Plots')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
def plot_confusion_matrix(y_test, y_pred, classes=False,title='Confusion Matrix', figsize=(5, 5),
                          label_size=11, title_size=14, text_size=12, savefig=False,
                          save_path='./'):
    '''
    Plot confusion matrix.
    '''
    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize our confusion matrix
    n_classes = cm.shape[0]
    
    # Pretify it
    fig, ax = plt.subplots(figsize=figsize)
    # Create a matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    # Create classes    
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    
    # Label the axes
    ax.set(title=title, xlabel='Predicted Label', ylabel='True Label',
           xticks=np.arange(n_classes), yticks=np.arange(n_classes),
           xticklabels=labels, yticklabels=labels)
    
    # Set x-axis labels to bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    
    # Adjust label size
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.title.set_size(title_size)
    
    # Change the plot x-labels vertically
    plt.xticks(rotation=70)
    
    # Set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.
    
    # Plot the text on each cell
    for i, j in itertools.product(range(n_classes), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)',
                 horizontalalignment='center', size=text_size,
                 color='white' if cm[i, j] > threshold else 'black')
    
    # Save the figure
    if savefig:
        fig.savefig(save_path + title + '.png')
        
def plot_f1_scores(f1_scores, figsize=(10, 20), title='F1-scores'):
    '''
    Plot a classification report items.
    '''
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(f1_scores)), f1_scores['f1_scores'].values)
    ax.set_yticks(range(len(f1_scores)))
    ax.set_yticklabels(f1_scores['class_names'])
    ax.set_xlabel('F1-score')
    ax.set_title(title)
    ax.invert_yaxis()
    plt.xticks(rotation=75)
    plt.show()

def compare_two_model_results(results_1, results_2):
    '''
    Compare and logs two model results.
    '''
    print('\nComparing two model results...')
    similar = (results_1 == results_2)
    conclusion = f'The two results are {"similar" if (similar) else "different"}'
    if not similar:
        # Check to see if the loaded weights model results are very close to the previous
        # non-loaded model results
        is_close = np.isclose(np.array(results_1), np.array(results_2))
        if len(is_close.shape) < 2:
            is_close = np.expand_dims(is_close, axis=0)
        is_close = any(any(row) for row in is_close)
        if is_close:
            # Check the difference between the two results
            difference_between_two_results = np.array(results_1) - np.array(results_2)
            conclusion = conclusion + ', but they are close. They have this much a difference:'
            conclusion = conclusion + f'\n{difference_between_two_results}'
        else:
            conclusion = conclusion + ' and not close.'
    print(conclusion)

