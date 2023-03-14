## Helpful Function ##

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import random

def plot_sample_images(data=[], figsize=(6, 6), cmap=plt.cm.binary, 
                       label_size=6, dir_path='', read_from_dir=False):
    '''
    Plotting image samples.
    '''
    print('\nCreating images sample plot')
    labels = [label.split('/')[0] for label in data]
    plt.figure(figsize=figsize)
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        rand_index = random.choice(range(len(data)))
        if read_from_dir:
            file = data[rand_index]
            img = mimg.imread(f'{dir_path}/{file}')
            plt.imshow(img)
        else:
            plt.imshow(data[rand_index], cmap=cmap)
        plt.title(labels[rand_index])
        ax.title.set_size(label_size)
        plt.axis(False)
    plt.show()

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
    
def plot_augmented_data(augmented_data, original_data=[], figsize=(6, 6), label_size=8):
    '''
    Plotting augmented image data.
    '''
    print('\nPlotting augmented image')
    augmented_images, augmented_labels = augmented_data.next()
    rand_index = random.choice(range(len(augmented_images)))
    augmented_img = augmented_images[rand_index]
    
    plt.figure(figsize=figsize)
    if len(original_data) > 0:
        ax = plt.subplot(1, 2, 1)
        plt.imshow(augmented_img)
        plt.axis(False)
        plt.title('Augmented Image')
        ax.title.set_size(label_size)
        
        og_images, og_labels = original_data.next()
        ax = plt.subplot(1, 2, 2)
        plt.imshow(og_images[rand_index])
        plt.axis(False)
        plt.title('Original Image')
        ax.title.set_size(label_size)
    else:
        plt.imshow(augmented_img)
        plt.axis(False)
        plt.title('Augmented Image')
    plt.show()
    
def load_and_preprocess_image(img_path='./', target_size=(224, 224)):
    '''
    Read an image from a filename, turns it into a tensor and reshapes it to
    target_size.
    '''
    # Read in the image
    img = tf.io.read_file(img_path)
    # Decode the read image file into a tensor
    img = tf.image.decode_image(img)
    # Resize the image
    img = tf.image.resize(img, size=target_size)
    # Rescale the image
    img = img/225.
    return img

def predict_and_visualize(model, labels=[], img_path='./image.jpg', 
                          figsize=(6, 6), title_size=10):
    img_data = load_and_preprocess_image(img_path, target_size=(64, 64))
    pred = model.predict(tf.expand_dims(img_data, axis=0))
    if len(pred[0]) > 1:
        print(tf.argmax(pred[0]))
        pred_class = labels[tf.argmax(pred[0])]
    else:
        pred_class = labels[int(tf.round(pred[0]))]
    print(f"\nPrediction on {img_path.split('/')[-1]}: {pred_class}")
    
    img = mimg.imread(img_path)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(f'prediction: {pred_class}')
    plt.axis(False)
    plt.show()
    
    