import pandas as pd
import numpy as np

from config import *

def load_train() -> pd.DataFrame:
    train = pd.read_csv(
        TRAIN_PATH,
        dtype={
            'user_id': np.int64,
            'book_id': np.int64,
            'has_read': np.int64,
            'rating': np.float64,
            'timestamp': np.str
        }
    )

    return train

def load_book_description() -> pd.DataFrame:
    data = pd.read_csv(
        BOOK_DESCRIPTIONS_PATH,
        dtype={
            'book_id': np.int64,
            'description': np.str
        }
    )

    return data

def load_book_genres() -> pd.DataFrame:
    data = pd.read_csv(
        BOOK_GENRES_PATH,
        dtype={
            'book_id': np.int64,
            'genre_id': np.int64
        }
    )

    return data

def load_books() -> pd.DataFrame:
    data = pd.read_csv(
        BOOKS_PATH,
        dtype={
            'book_id': np.int64,
            'title': np.str,
            'author_id': np.int64,
            'author_name': np.str,
            'publication_year': np.int64,
            'language': np.int64,
            'avg_rating': np.float64,
            'publisher': np.float64
        }
    )

    return data

def load_genres() -> pd.DataFrame:
    data = pd.read_csv(
        GENRES_PATH,
        dtype={
            'genre_id': np.int64,
            'genre_name': np.str,
            'books_count': np.int64
        }
    )

    return data

def load_test() -> pd.DataFrame:
    data = pd.read_csv(
        TEST_PATH,
        dtype={
            'user_id': np.int64,
            'book_id': np.int64
        }
    )

    return data

def load_users() -> pd.DataFrame:
    data = pd.read_csv(
        USERS_PATH,
        dtype={
            'user_id': np.int64,
            'gender': np.int64
            'age': np.int64
        }
    )

    return data


if __name__ == '__main__':
    ...