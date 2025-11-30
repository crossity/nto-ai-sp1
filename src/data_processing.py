from src.data_load import *

def process_books() -> pd.DataFrame:
    books = load_books()
    books_genres = load_book_genres()
    genres = pd.read_parquet(LOCAL_DATA_PATH / 'genres.pq')

    books_genres = books_genres.merge(genres, on='genre_id', how='left')
    books_genres = books_genres.drop('genre_id', axis=1)

    agg_type = {
        'books_count': 'median'
    }
    for col in books_genres.filter(like='vec').columns:
        agg_type[col] = 'mean'

    books_genres = books_genres.groupby('book_id').agg(agg_type).reset_index()

    books = books.merge(books_genres, on='book_id', how='left')

    return books

def process_fav_genre() -> pd.DataFrame:
    train = load_train()[['user_id', 'book_id', 'rating']]
    users = load_users()[['user_id']]
    books = pd.read_parquet(LOCAL_DATA_PATH / 'books.pq')

    vec_cols = books.filter(like='vec').columns

    books = books.set_index('book_id')[vec_cols]

    X = train.join(books, on='book_id', how='left')

    X = X.dropna(subset=[vec_cols[0]]) 

    weighted = X[vec_cols].multiply(X['rating'], axis=0)

    num = weighted.groupby(X['user_id'], sort=False).sum()
    den = X.groupby('user_id', sort=False)['rating'].sum()

    fav = num.div(den, axis=0)

    fav = users.set_index('user_id').join(fav, how='left').fillna(0.0)

    fav.columns = [f'fav_{c}' for c in vec_cols]

    fav = fav.reset_index()

    fav[ fav.columns.difference(['user_id']) ] = fav[ fav.columns.difference(['user_id']) ].astype('float32')

    return fav


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
    # print('Generating books tables...')
    # books = process_books()
    # books.to_parquet(LOCAL_DATA_PATH / 'books.pq', index=False)

    ### Generating user favourite genre
    fav_genres = process_fav_genre()
    fav_genres.to_parquet(LOCAL_DATA_PATH / 'fav_genres.pq', index=False)

    ## Ungrouping training data
    print('Ungrouping train data...')
    train = load_train()
    train = process_ratings(train)

    print('Saving train...')
    train.to_parquet(LOCAL_DATA_PATH / 'train.pq', index=False)

    ## Ungrouping testing data
    print('Ungrouping test data...')
    test = load_test()
    test = process_ratings(test)

    print('Saving test...')
    test.to_parquet(LOCAL_DATA_PATH / 'test.pq', index=False)


if __name__ == '__main__':
    process()
