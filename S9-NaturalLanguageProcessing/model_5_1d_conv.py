## Building Model 5: 1D-Convolutional Neural Network (CNN) ##

import pandas as pd
import random as rand
import tensorflow as tf
import my_utils as utils

print(f'\ntensorflow: {tf.__version__}')

DF_PATH = '../data/nlp_getting_started'
SAVE_DIR = './tmp/model_logs'

MAX_VOCAB_LENGTH = 10000 # Max number of words to have in our vocabularies
MAX_LENGTH = 15 # Max length of our sequences will be (e.g. How many words from Tweet does a model see?)

def run():
    '''
    Execute the code.
    '''
    
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
    
    # Visualizing the data
    print('\nVisualizing some random data: ')
    for _ in range(4):
        rand_idx = rand.randint(0, len(training_texts))
        text = training_texts[rand_idx]
        label = training_labels[rand_idx]
        print(f'Target: {label}', '(diaster)' if label == 1 else '(not real diaster)')
        print(f'Text:\n{text}')
        print('\n---\n')
    
    ## Text vectorization
    from tensorflow.keras.layers import TextVectorization
    
    # Use the default TextVectorization parameters
    text_vectorizer = TextVectorization(max_tokens=MAX_VOCAB_LENGTH,
                                        output_mode='int',
                                        output_sequence_length=MAX_LENGTH)
    
    # Fit the text vectorizer to the training set
    text_vectorizer.adapt(training_texts)
    
    # Choose a random text from training dataset and tokenize it
    random_text = rand.choice(training_texts)
    print(f'\nOriginal text: \n{random_text}\
          \n\nVectorized version: \n', text_vectorizer([random_text]))
    
    # Get the unique words in the vocabulary
    words_in_vocab = text_vectorizer.get_vocabulary() # Get all the unique words in our vocabulary
    top_5_words = words_in_vocab[:5]
    bottom_5_words = words_in_vocab[-5:]
    print(f'\nNumber of words in vocabulary: {len(words_in_vocab)}')
    print(f'5 most common words: {top_5_words}')
    print(f'5 least common words: {len(bottom_5_words)}')
    
    ## Creating an Embedding using Embedding Layer
    from tensorflow.keras.layers import Embedding
    
    embedding = Embedding(input_dim=MAX_VOCAB_LENGTH, # Set input shape
                          output_dim=128, # Set output shape
                          input_length=MAX_LENGTH) # How long is each input
    
    # Get a random text from the training set
    random_text_2 = rand.choice(training_texts)
    print(f'\nOriginal text: \n{random_text_2}')
    
    # Embed the random text (turn it into dense vector of fixed size)
    embedded_random_text_2 = embedding(text_vectorizer([random_text_2]))
    print('\nEmbedded version: \n', embedded_random_text_2)
    
    ## Create the Model 5: 1D-Convolutional Neural Network (CNN)
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPool1D, Dense
    from tensorflow.keras.activations import relu, sigmoid
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import BinaryCrossentropy
    
    # Test out our embedding, Conv1D, and max pooling layers
    embedding_test = embedding(text_vectorizer(['This is a test text sample.'])) # Turn target sequence into embedding
    conv_1d = Conv1D(filters=32,
                     kernel_size=5, # This is also referred to as an ngram of 5 (meaning it looks at 5 words at a time)
                     strides=1, # default
                     activation=relu,
                     padding='valid') # defualt="valid", the output is smaller than the input shape, "same" means the output is same
    conv_1d_output = conv_1d(embedding_test) # Pass test embedding through conv1d layer
    max_pool = GlobalMaxPool1D()
    max_pool_output = max_pool(conv_1d_output) # Equivalent to "get the most important feature" or "get the feature with the highest value"
    
    print('\nEmbedding output:\n', embedding_test)
    print('\nConv1D output:\n', conv_1d_output)
    print('\nGlobalMaxPool1D output:\n', max_pool_output)
    
    '''
    NOTE:
    For different explanations of parameters see:
    *   https://poloclub.github.io/cnn-explainer/ (this is for 2D but can relate to 1D data)
    *   Different between "same" and "valid" paddings: 
        https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t/
    '''
    
    # Creating Model 5: Conv1D
    inputs = Input(shape=(1,), dtype=tf.string, name='input_layer')
    x = text_vectorizer(inputs)
    x = embedding(x)
    x = Conv1D(filters=64,
               kernel_size=5,
               strides=1,
               activation=relu,
               padding='valid')(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(64, activation=relu)(x)
    outputs = Dense(1, activation=sigmoid)(x)
    model_5 = Model(inputs, outputs, name='model_5_conv1d')
    
    # Compiling the model
    model_5.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
    
    # Model summary
    print('\nModel Summary:')
    model_5.summary()
    
    # Fitting the model
    print(f'\nFitting {model_5.name}')
    total_20_percent = int(.2 * len(training_texts))
    validation_texts_20_percent = validation_texts[:total_20_percent]
    validation_labels_20_percent = validation_labels[:total_20_percent]
    model_5_tensorboard_cb = utils.create_tensorboard_callback(dir_name=SAVE_DIR, 
                                                               experiment_name=model_5.name)
    model_5_history = model_5.fit(training_texts, training_labels,
                                  epochs=5,
                                  validation_data=(validation_texts_20_percent,
                                                   validation_labels_20_percent),
                                  callbacks=[model_5_tensorboard_cb])
    
    ## Evaluate the model
    print(f'\nEvaluating {model_5.name}')
    model_5.evaluate(validation_texts, validation_labels)
    utils.plot_model_history(model_5_history)
    
    ## Making predictions & compare the results
    print(f'\nMaking predictions using {model_5.name}')
    model_5_preds = model_5.predict(validation_texts)
    model_5_preds = tf.squeeze(tf.round(model_5_preds))
    model_5_results = utils.evaluate_model_results(validation_labels, model_5_preds)
    print(f'\nModel 5 (Conv1D) results: \n{model_5_results}')
    utils.plot_confusion_matrix(validation_labels, model_5_preds)

    return model_5_history, model_5_results

## Run the file
if __name__ == '__main__':
    run()
