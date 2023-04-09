## Building Model 2: LSTM Model (RNN) ##

'''
NOTE:
Recurrent Neural Networks (RNNs)
RNNs are usefule for sequence data. The premise of recurrent neural network is to
use the representation of the previous input to add the representation of a later
input. If you want an overview of the internals of a recurrent neural networks,
see the following:
*   MIT's sequence modeling: https://youtu.be/qjrad0V0uJE
*   Chris Olah's intro to LSTMs: https://colah.github.io/posts/2015-08-Understanding-LSTMs
*   Andrej Karpathy's the unreasonable effectiveness of of recurrent neural networks:
    http://karpathy.github.io/2015/05/21/rnn-effectiveness
'''

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
    
    ## Create the Model 2: LSTM
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, LSTM, Dense
    from tensorflow.keras.activations import relu, sigmoid
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import BinaryCrossentropy
    
    '''
    NOTE:
    LSTM = Long-Short Term Memory (one of the most popular LSTM layers)
    The structure of an RNN typically looks like this:
    Input (text) -> Tokenize -> Embedding -> Layers (RNN/Dense) -> Output (label probability)
    '''
    
    # Building the model with Keras Functional API
    inputs = Input(shape=(1,), dtype=tf.string, name='input_layer')
    x = text_vectorizer(inputs)
    x = embedding(x)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(64)(x)
    x = Dense(64, activation=relu)(x)
    outputs = Dense(1, activation=sigmoid, name='output_layer')(x)
    model_2 = Model(inputs, outputs, name='model_2_lstm')
    
    # Compiling the model
    model_2.compile(optimizer=Adam(),
                    loss=BinaryCrossentropy(),
                    metrics=['accuracy'])
    
    # Modal summary
    print('\nModel SUmmary:')
    model_2.summary()
    
    # Fitting the model
    print(f'\nFitting {model_2.name}')
    total_20_percent = int(.2 * len(training_texts))
    validation_texts_20_percent = validation_texts[:total_20_percent]
    validation_labels_20_percent = validation_labels[:total_20_percent]
    model_2_tensorboard_cb = utils.create_tensorboard_callback(dir_name=SAVE_DIR, 
                                                               experiment_name=model_2.name)
    model_2_history = model_2.fit(training_texts, training_labels,
                                  epochs=5,
                                  validation_data=(validation_texts_20_percent,
                                                   validation_labels_20_percent),
                                  callbacks=[model_2_tensorboard_cb])
    
    ## Evaluating the model
    print(f'\nEvaluating {model_2.name}')
    model_2.evaluate(x=validation_texts, y=validation_labels)
    utils.plot_model_history(model_2_history)
    
    ## Making predictions & compare the results
    print(f'\nMaking predictions using {model_2.name}')
    model_2_preds = model_2.predict(validation_texts)
    model_2_preds = tf.squeeze(tf.round(model_2_preds))
    model_2_results = utils.evaluate_model_results(validation_labels, model_2_preds)
    print(f'\nModel 2 (LSTM) results: \n{model_2_results}')
    utils.plot_confusion_matrix(validation_labels, model_2_preds)
    
    return model_2_history, model_2_results

## Run the file
if __name__ == '__main__':
    run()
