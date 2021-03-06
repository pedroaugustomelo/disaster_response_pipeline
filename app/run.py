import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin



app = Flask(__name__)

def tokenize(text):
    '''
    Tokenize, lemmatize, normalize, and remove stop words from the input
    Args: 
        text: string of words
    Output: 
        clean tokens: cleaned words
     
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))

    # Get clean tokens after lemmatization, normalization, stripping and stop words removal
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if tok not in stopWords:
            clean_tokens.append(clean_tok)

    return clean_tokens

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

# load data
engine = create_engine('sqlite:///../data/disasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    multiple_labels = df.drop(columns=['id','message','original','genre']).sum(axis=1)
    positive_labels = df.drop(columns=['id','message','original','genre']).sum()
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Histogram(
                    x=multiple_labels
                )
            ],

            'layout': {
                'title': 'Histogram of Multiple Lables Instances',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "# of Multiple Lables"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=positive_labels.index,
                    y=positive_labels
                )
            ],

            'layout': {
                'title': 'Distribution of positive lables',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Output Lables"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()