# preprocessing.py

import pandas as pd

def load_data(ratings_path='ratings.csv', movies_path='movies.csv'):
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    return ratings, movies

def preprocess_data(ratings, movies):
    # Remove duplicates
    ratings.drop_duplicates(inplace=True)
    movies.drop_duplicates(inplace=True)

    # Drop missing values
    ratings.dropna(inplace=True)
    movies.dropna(inplace=True)

    # Drop unnecessary columns
    if 'timestamp' in ratings.columns:
        ratings.drop('timestamp', axis=1, inplace=True)

    # Standardize text: lowercase titles and genres
    movies['title'] = movies['title'].str.lower()
    movies['genres'] = movies['genres'].str.lower()

    # Replace "(no genres listed)" with empty string
    movies['genres'] = movies['genres'].replace('(no genres listed)', '')

    # Merge ratings and movie metadata
    data = pd.merge(ratings, movies, on='movieId')

    return data