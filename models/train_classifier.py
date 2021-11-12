import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import time
import pickle

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report


def load_data(database_filepath):
     ''' Load data from database, create two datasets: one for independent
         variable and other for labels.
         Args:
              database_filepath: path of database
         Output:
              X : dataset of independent variable
              y : dataset of labels
     '''
     engine = create_engine('sqlite:///' + database_filepath)
     df = pd.read_sql_table('Messages', engine)
     X = df['message']
     y = df.iloc[:, 5:39]
     
     return X,y, y.columns

def tokenize(text):
    ''' tokenize and lemmatize text.
    args: 
        text: string to tokenize
    output:
        clean_tokens: list of lemmatized tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if tok not in stopWords:
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
      
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('text_len', TextLengthExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
    ])
    parameters = {
        
    'clf__estimator__n_estimators' : [50, 100],
    'clf__estimator__max_depth' : [10, 100]
     }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)
    # Turn prediction into DataFrame
    y_pred = pd.DataFrame(y_pred,columns= y.columns)
    # For each category column, print performance
    for col in y.columns:
    print(f'Column Name:{col}\n')
    print(classification_report(Y_test[col],y_pred[col]))


def save_model(model, model_filepath):
    
    pickle.dump(model, open(filename, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()