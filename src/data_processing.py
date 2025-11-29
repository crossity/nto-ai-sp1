from src.data_load import *

def process_books() -> pd.DataFrame:
    books = load_books()
    books_genres = load_book_genres()
    genres = load_genres()

    books_genres = books_genres.merge(genres, on='genre_id', how='left')
    books_genres = books_genres.drop('genre_id', axis=1)

    books_genres = books_genres.groupby('book_id').agg({
        'books_count': 'mean',
        'genre_name': lambda x: '|'.join(x)
    }).reset_index()

    books = books.merge(books_genres, on='book_id', how='left')

    return books

def process_ratings(data: pd.DataFrame) -> pd.DataFrame:
    # train = load_train()

    # print(train)

    ## Replacing user id with user info
    print('Loading user info...')
    users = load_users()

    data = data.merge(users, on='user_id', how='left')
    # data = data.drop('user_id', axis=1)

    ## Replacing book id with book info
    print('Loading books info...')
    books = pd.read_parquet(LOCAL_DATA_PATH / 'books.pq')

    data = data.merge(books, on='book_id', how='left')
    # data = data.drop('book_id', axis=1)   

    ## Adding descriptions
    print('Loading descriptions...')
    descriptions = pd.read_parquet(LOCAL_DATA_PATH / 'descriptions.pq')

    print('Merging descriptions...')
    data = data.merge(descriptions, on='book_id', how='left')

    ## Removing useless data
    data = data.drop('author_name', axis=1)

    return data

def process():
    LOCAL_DATA_PATH.mkdir(parents=True, exist_ok=True)

    ## Generating books grouped tables
    print('Generating books tables...')
    books = process_books()
    books.to_parquet(LOCAL_DATA_PATH / 'books.pq', index=False)

    ## Ungrouping training data
    print('Ungrouping train data...')
    train = load_train()
    train = process_ratings(train)

    print('Saving train...')
    train.to_parquet(LOCAL_DATA_PATH / 'train.pq', index=False)


if __name__ == '__main__':
    process()
