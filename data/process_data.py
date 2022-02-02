import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    '''load data into pandas dataframes and merge both
    Args: 
       messages_filepath: file path for message csv
       categories_filepath: file path for categories csv
    Output:
       df: merged df of both dfs
    '''
    #load messages data
    messages = pd.read_csv(messages_filepath)
    #load categories data
    categories = pd.read_csv(categories_filepath)
    #merge messages and categories data 
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    ''' cleans data: creates categories dummies and drops original colum, remove duplicates. 
    Args:
        df: merged df of messages and categories
    Output: 
        df: cleaned df
    '''
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df.categories.str.split(';',expand=True))
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories
    category_colnames = row.str.split('-').apply(lambda x:x[0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
          # set each value to be the last character of the string
          categories[column] = categories[column].astype(str).str[-1:]
        
          #convert all categories to binary
          categories[column] = categories[column].astype('str').replace('2', '1')
    
          # convert column from string to numeric
          categories[column] = categories[column].astype('int32')
    
    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, left_index=True, right_index=True)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df
   

def save_data(df, database_filename):
    ''' Create Database and insert dataframe on it.
    args: 
        df: cleaned dataframe
        database_filename: database name
    '''
    # Create database engine
    engine = create_engine('sqlite:///'+database_filename)
    # Save df to database
    df.to_sql('disaster_response', engine, if_exists = 'replace', index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
