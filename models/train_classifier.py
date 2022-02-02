import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import joblib


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
from sklearn.base import BaseEstimator, TransformerMixin
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
     #read sql table
     engine = create_engine('sqlite:///' + database_filepath)
     df = pd.read_sql_table('disaster_response', engine)
    
     # Split features and targets
     X = df['message']
     y = df.drop(columns=['id','message','original','genre'])
     
     return X,y, y.columns

class TextLength(BaseEstimator, TransformerMixin):
    '''
    Overrides BaseEstimator and TransformerMixin to return text length

    '''
    def fit(self, X, y = None):
        '''
        Return self
        '''
        return self
    def transform(self, X):
        '''
        Count the text length on each part of X
        '''
        x_length = pd.Series(X).str.len()
        return pd.DataFrame(x_length)
    


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
    #lemmatize each token and remove stopwords 
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if tok not in stopWords:
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    ''' Builds pipeline to transform data and uses RandomForestClassifier as estimator.
        Tests multiple hyperparameters to tune the model with GridSearch.
        
        Output:
               cv: GridSearch for pipeline
    '''
    #Build pipeline with one Feature Union in order to merge togheter the numeric transformations. 
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('text_len', TextLength())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
    ])
    # parameters to tune the model 
    parameters = {
        
    'clf__estimator__n_estimators' : [50, 100],
    'features__text_pipeline__tfidf__use_idf': (True, False)
     }
    #Cross Validation on pipeline to find best parameters
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    ''' Evaluates model comparing prediction with test set: prints precision, recall, f1-score and support
        about each column.
        Args: 
             model: cv used to predict
             X_test: feature test dataset
             Y_test: target test dataset
             category_names: columns to compare
    '''     
    # predict test set  
    y_pred = model.predict(X_test)
    # Turn prediction into DataFrame
    y_pred = pd.DataFrame(y_pred,columns= category_names)
    # For each category column, print report
    for col in category_names:
        print(f'Column Name:{col}\n')
        print(classification_report(Y_test[col],y_pred[col]))


def save_model(model, model_filepath):
    '''save model as a pickle file.
    Args: 
         model: trained model
         model_filepath: path of model
    '''
    joblib.dump(model, model_filepath)

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