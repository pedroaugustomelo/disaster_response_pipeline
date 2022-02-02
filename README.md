# Disaster Response Pipeline Project
### Summary 
In this project, we received data from Figure Eight in order to build an web app able to classify new entries.

The data set represents messages sent during disasters and its labels, separated in 36 categories. First, we cleaned the data and saved it in a SQLite Database. Then we used it to train a Machine Learning Pipeline, exporting the model to classify the input provided on the Flask Web App.

Finally, we used the data to plot some visualization in the web app and an input to predict the categories of new messages, like shown bellow:

![Captura de Tela 2022-02-01 às 22 55 59](https://user-images.githubusercontent.com/16378223/152081461-56c89fa9-ece2-4872-af9f-312f9b053e2d.png)
![Captura de Tela 2022-02-01 às 22 55 44](https://user-images.githubusercontent.com/16378223/152081468-350b9527-ebde-4370-bdbc-ff76113d5fe8.png)
![Captura de Tela 2022-02-01 às 22 56 21](https://user-images.githubusercontent.com/16378223/152081572-fe5ad1b5-0ef3-4993-bce2-2b4963c8cf1c.png)

### Project Components

Our project is divided in:

1. ETL Pipeline
In process_data.py, we implemented a data cleaning pipeline that:

Loads the data sets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

2. Machine Learning Pipeline
In train_classifier.py, it has a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

3. Flask Web App
In this part, we output the visualizations of the original data and classify the sentence provided by the client, across 36 categories.

### Installations

- Python 3.9
- Pandas
- NumPy
- Sklearn
- Nltk
- SQLAlchemy
- Joblib
- Bootstrap
- Flask
- Plotly

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing, Authors, Acknowledgements

This app was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
