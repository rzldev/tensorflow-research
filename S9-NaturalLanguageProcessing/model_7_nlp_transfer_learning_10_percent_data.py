## Building Model 7: TensorFlow Hub Pretrained Feature Extractor ##
## (Transfer Learning for NLP) with 10% Training Data ##

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
    
    ## Creating Model 7: TF Hub Pretrained USE but with 10% of training data
    import tensorflow_hub as hub
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.activations import relu, sigmoid
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import BinaryCrossentropy
    
    # Prepare the data
    '''
    NOTE:
    Be very careful when creating training/validation/test splits that you don't leak
    the data accross the dataset, otherwise your model evaluation metrics will be wrong.
    If something looks too good to be true (a model trained on 10% of the data outperforming
    the model which trained on 100% data), trust your gut and go back through to find
    where the error may lie.
    '''
    
    _, training_texts_10_percent, _, training_labels_10_percent = train_test_split(training_texts, 
                                                                                   training_labels,
                                                                                   test_size=.1,
                                                                                   random_state=42)
    
    # Comparing the training labels and 10% traiing labels
    print(f'\nTraining labels:\n{pd.Series(training_labels).value_counts()}')
    print(f'\n10% training labels:\n{pd.Series(training_labels_10_percent).value_counts()}')
    
    # Testing out TensorFlow's Hub Universal Sentence Encoder 4
    sample_text = 'Google is one of the biggest tech company.'
    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    embedded_samples = embed([sample_text, 'Rendang is one of the most tastiest food i\'ve known.'])
    print('\nTensorFlow\'s Hub Unversal Sentence Encoder embedded samples:\n', embedded_samples)
    
    # Create a Keras Layer using USE (Universal Sentence Encoder) pretrained layer from TensorFlow Hub
    sentence_encoder_layer = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4',
                                            input_shape=[],
                                            dtype=tf.string,
                                            trainable=False,
                                            name='use_layer')
    
    # Create Model 7: Transfer Learning for NLP
    model_7 = Sequential(name='model_7_use')
    model_7.add(sentence_encoder_layer)
    model_7.add(Dense(64, activation=relu))
    model_7.add(Dense(1, activation=sigmoid))
    
    # Compiling the model
    model_7.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
    
    # Model summary
    print('\nModel Summary:')
    model_7.summary()
    
    # Fitting the model
    print(f'\nFitting {model_7.name}')
    total_20_percent = int(.2 * len(training_texts))
    validation_texts_20_percent = validation_texts[:total_20_percent]
    validation_labels_20_percent = validation_labels[:total_20_percent]
    model_7_tensorboard_cb = utils.create_tensorboard_callback(dir_name=SAVE_DIR, 
                                                               experiment_name=model_7.name)
    model_7_history = model_7.fit(training_texts_10_percent, training_labels_10_percent,
                                  epochs=5,
                                  validation_data=(validation_texts_20_percent,
                                                   validation_labels_20_percent),
                                  callbacks=[model_7_tensorboard_cb])
    
    # Evaluate the model
    print(f'\nEvaluating {model_7.name}')
    model_7.evaluate(validation_texts, validation_labels)
    utils.plot_model_history(model_7_history)
    
    # Making predictions & compare the results
    print(f'\nMaking predictions using {model_7.name}')
    model_7_preds = model_7.predict(validation_texts)
    model_7_preds = tf.squeeze(tf.round(model_7_preds))
    model_7_results = utils.evaluate_model_results(validation_labels, model_7_preds)
    print(f'\nModel 7 (Universal Sentence Encoder) results: \n{model_7_results}')
    utils.plot_confusion_matrix(validation_labels, model_7_preds)
    
    return model_7_history, model_7_results

## Run the file
if __name__ == '__main__':
    run()
