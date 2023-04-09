## Building Model 0: Naive Bayes (baseline) ## 

'''
NOTE:
As with all machine learning modeling experiments, it's important to create a baseline
model so we got a benchmark for future experiments to build upon. To create our
baseline, we'll use Sklearn's Multinominal Naive Bayes using the TF-IDF formula to
convert our words to numbers.

It's common practice to use non-DL algorithms as a baseline because of their speed
and then later using DL to see if you can improve upon them.
'''

import pandas as pd
import random as rand
import my_utils as utils

DF_PATH = '../data/nlp_getting_started'

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
    
    ## Building the model
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    
    # Create tokenization and modelling pipeline
    model_0 = Pipeline([
        ('tfidf', TfidfVectorizer()), # Convert words to numbers using tfidf
        ('clf', MultinomialNB()) # Model the text
        ])
    model_0.name = 'model_0_baseline'
        
    # Fit the pipeline to the training data
    print(f'\nFitting {model_0.name}')
    model_0.fit(training_texts, training_labels)
    
    ## Evaluate the baseline model
    print(f'\nEvaluating {model_0.name}')
    model_0_score = model_0.score(validation_texts, validation_labels)
    print(f'\nModel 0 (Naive Bayes) score: {model_0_score}')
    
    ## Making predictions & compare the results
    model_0_preds = model_0.predict(validation_texts)
    model_0_results = utils.evaluate_model_results(validation_labels, model_0_preds)
    print('\nModel 0 (Naive Bayes) results:\n', model_0_results)
    
    return model_0, model_0_results

## Run the file
if __name__ == '__main__':
    run()
