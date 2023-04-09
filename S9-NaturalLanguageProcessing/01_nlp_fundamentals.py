## NLP Fundamentals With TensorFlow ##

'''
NOTE:
NLP has the goal of deriving information out of natural language (could be sequences
text or speech). Another common term for NLP problems is to sequence problems (seq2seq).
'''
import pandas as pd
import random as rand
import tensorflow as tf

print(f'\ntensorflow: {tf.__version__}')

DF_PATH = '../data/nlp_getting_started'

## Get a text dataset
'''
NOTE:
The dataset we are going to be using is Kaggle's introduction to NLP dataset (text
samples of Tweets labelled as disaster or not disaster). See the original source
here: https://kaggle.com/c/nlp-getting-started. You can get the data through this
link here: https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip
'''

# Importing the dataset
training_set = pd.read_csv(f'{DF_PATH}/train.csv')
test_set = pd.read_csv(f'{DF_PATH}/test.csv')
print('\nTraining set:\n', training_set.head())
print('\nTest set:\n', test_set.head())

# Splitting the training set into training and validation set
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
    
## Converting text into numbers
'''
NOTE:
When dealing with a text problem, one of the first things you'll have to do before
you can build a model in to convert your text into number. There are few ways to
do this, namely:
*   Tokenization
    Direct mapping of token (a token could be word or character) to number.
*   Embedding
    Create a matrix of feature vector for each token (the size of feature vector
    can be defined and this embedding can be learned).
'''

## Text vectorization
from tensorflow.keras.layers import TextVectorization

# Use the default TextVectorization parameters
text_vectorizer = TextVectorization(max_tokens=None, # How many words in the vocabulary (automatically add <OOV>)
                                    standardize='lower_and_strip_punctuation',
                                    split='whitespace',
                                    ngrams=None, # Create a groups of n-words?
                                    output_mode='int', # How to map tokens to number
                                    output_sequence_length=None, # How long do you want your sequences to be?
                                    pad_to_max_tokens=False)

# Find the average number of tokens (words) in the training tweets
print(training_texts[0].split())
print(round(sum([len(i.split()) for i in training_texts])/len(training_texts)))

# Setup TextVectorization variable
max_vocab_length = 10000 # Max number of words to have in our vocabularies
max_length = 15 # Max length of our sequences will be (e.g. How many words from Tweet does a model see?)
text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode='int',
                                    output_sequence_length=max_length)

# Fit the text vectorizer to the training set
text_vectorizer.adapt(training_texts)

# Create a sample text and tokenize it
sample_text = 'I \'m an engineer'
print(f'\nOriginal text: \n{sample_text}\
      \n\nVectorized version: \n', text_vectorizer([sample_text]))
      
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
'''
NOTE:
To make our Embedding, we are going to use TensorFlow's Embedding Layer. You can
look up this source to find more about it: 
https://tensorflow.org/api_docs/python/tf/keras/layers/Embedding

The parameters we care most about for our embedding layer:
*   `input_dim`:    The size of our vocabulary.
*   `output_dim`:   The size of the output embedding vector, for example a value of
                    100 would mean each token gets represented by a vector 100 long.
*   `input_length`: Length of the sequences being passed to the embedding layer.
'''

embedding = Embedding(input_dim=max_vocab_length, # Set input shape
                      output_dim=128, # Set output shape
                      input_length=max_length) # How long is each input

# Get a random text from the training set
random_text_2 = rand.choice(training_texts)
print(f'\nOriginal text: \n{random_text_2}')

# Embed the random text (turn it into dense vector of fixed size)
embedded_random_text_2 = embedding(text_vectorizer([random_text_2]))
print('\nEmbedded version: \n', embedded_random_text_2)

# Check out a single token's embedding
print('\nSingle token embedding:\n', embedded_random_text_2[0][0], '\n', random_text_2)

## Modeling on a text dataset (running a series of experiments)
'''
NOTE:
No we've got a way to turn our text sequences into numbers, it's time to start building
a series of modelling experiments. We'll put each of the experiments in separate files.
We'll start with a baseline and will move on from there:
*   Model 0: Naive Bayes (baseline), this is from sklearn ML map:
    https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
*   Model 1: Feed-Forward Neural Network (dense model)
*   Model 2: LSTM model (RNN)
*   Model 3: GRU model (RNN)
*   Model 4: Bidirectional-LSTM model (RNN)
*   Model 5: 1D-Convolutional Neural Network (CNN)
*   Model 6: TensorFlow Hub Pretrained Feature Extractor (Transfer Learning for NLP)
*   Model 7: Same as model 6 with 10% training data

How are we going to approach all of these? Use the standard steps in modeling with
TensorFlow:
1.  Create a model
2.  Build a model
3.  Fit the model
4.  Evaluate the model
'''
