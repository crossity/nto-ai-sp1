from src.data_load import *

def process_books():
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

    books.to_csv(LOCAL_DATA_PATH / 'books.csv', index=False)

def process(data: pd.DataFrame) -> pd.DataFrame:
    # train = load_train()

    # print(train)

    ## Replacing user id with user info
    users = load_users()

    data = data.merge(users, on='user_id', how='left')
    data = data.drop('user_id', axis=1)

    ## Replacing book id with book info
    books = pd.read_csv(LOCAL_DATA_PATH / 'books.csv')

    data = data.merge(books, on='book_id', how='left')
    data = data.drop('book_id', axis=1)

    ## Removing useless data
    data = data.drop('author_id', axis=1)

    return data


if __name__ == '__main__':
    # process_books()
    train = load_train()
    train = process(train)
    train.to_csv(LOCAL_DATA_PATH / 'train.csv', index=False)
