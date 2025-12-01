import numpy as np

from sklearn.decomposition import TruncatedSVD

from src.data_load import *


def _l2_normalize_df(mat: pd.DataFrame) -> pd.DataFrame:
    nrm = np.linalg.norm(mat.values, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return pd.DataFrame(mat.values / nrm, index=mat.index, columns=mat.columns)

def build_temporal_fav_genre(train_hist: pd.DataFrame,
                             books: pd.DataFrame,
                             alpha: float = 1.0,
                             tau_days: float = 120.0,
                             m_shrink: float = 3.0) -> pd.DataFrame:
    """
    Строит любимый жанровый профиль пользователя по ИСТОРИИ train_hist (без последних записей),
    с весами по рейтингу и давности, и shrinkage к глобальному prior.
    train_hist: user_id, book_id, rating, timestamp (<= t_last - eps)
    books: book_id + vec_0..vec_D (L2-нормализованные колонки 'vec_*')
    """
    vec_cols = [c for c in books.columns if c.startswith('vec_')]
    B = books.set_index('book_id')[vec_cols].astype('float32')
    B = _l2_normalize_df(B)  # на всякий случай

    # средний вектор по всем книгам как prior
    prior = B.mean(axis=0).astype('float32').values  # shape (D,)

    df = train_hist[['user_id', 'book_id', 'rating', 'timestamp']].copy()
    df = df.merge(B, left_on='book_id', right_index=True, how='left')

    # центрирование рейтинга по пользователю
    mu = df.groupby('user_id')['rating'].transform('mean')
    r_pos = (df['rating'] - mu).clip(lower=0)

    # экспоненциальное затухание (чем старее, тем меньше вес)
    ref_time = df.groupby('user_id')['timestamp'].transform('max')
    dt_days = ( pd.to_datetime(ref_time) -  pd.to_datetime(df['timestamp'])).dt.total_seconds() / (3600 * 24)
    w = (r_pos ** alpha) * np.exp(-dt_days / tau_days)

    for c in vec_cols:
        df[c] = df[c].astype('float32') * w.values.astype('float32')

    num = df.groupby('user_id', sort=False)[vec_cols].sum()
    den = w.groupby(df['user_id']).sum()

    # shrinkage к prior
    num = num.add(m_shrink * prior, axis='columns')
    den = den + m_shrink

    fav = num.div(den, axis=0).astype('float32')
    fav = _l2_normalize_df(fav)

    fav.reset_index(inplace=True)
    fav.columns = ['user_id'] + [f'fav_{c}' for c in vec_cols]
    # сила профиля — сколько фактов в истории (как фича)
    cnt = df.groupby('user_id').size().rename('fav_genre_cnt').reset_index()
    fav = fav.merge(cnt, on='user_id', how='left').fillna({'fav_genre_cnt': 0})

    return fav

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

def add_cosine_similarity(df_pairs: pd.DataFrame,
                          fav_users: pd.DataFrame,
                          books: pd.DataFrame) -> pd.DataFrame:
    user_vec_cols = [c for c in fav_users.columns if c.startswith("fav_vec_")]
    book_vec_cols = [c for c in books.columns if c.startswith("vec_")]

    print(fav_users.head())

    df = (df_pairs
          .merge(fav_users[['user_id', 'fav_genre_cnt'] + user_vec_cols], on='user_id', how='left')
          .merge(books[['book_id'] + book_vec_cols], on='book_id', how='left'))

    # нормализуем (на случай дрейфа)
    U = df[user_vec_cols].to_numpy(np.float32)
    B = df[book_vec_cols].to_numpy(np.float32)

    U_norm = np.linalg.norm(U, axis=1, keepdims=True); U_norm[U_norm == 0] = 1.0
    B_norm = np.linalg.norm(B, axis=1, keepdims=True); B_norm[B_norm == 0] = 1.0

    U_hat = U / U_norm
    B_hat = B / B_norm

    cos = np.sum(U_hat * B_hat, axis=1).astype('float32')
    df['genre_cosine'] = cos

    # Доп. фича: «сила» профиля
    df['genre_profile_strength'] = (U_norm.flatten()).astype('float32')
    df['genre_profile_cnt'] = df['fav_genre_cnt'].astype('float32').fillna(0.0)

    return df[['user_id', 'book_id', 'genre_cosine', 'genre_profile_strength', 'genre_profile_cnt']]

def add_lowdim_components(df_pairs, fav_users, books, k=16):
    user_vec_cols = [c for c in fav_users.columns if c.startswith("fav_vec_")]
    book_vec_cols = [c for c in books.columns if c.startswith("vec_")]

    # учим SVD на векторах книг (стабильное основание)
    svd = TruncatedSVD(n_components=k, random_state=42)
    Z_books = svd.fit_transform(books[book_vec_cols].to_numpy(np.float32))

    books_low = books[['book_id']].copy()
    books_low[[f'book_svd_{i}' for i in range(k)]] = Z_books

    # проектируем пользовательский профиль в то же пространство
    Z_users = svd.transform(fav_users[user_vec_cols].to_numpy(np.float32))
    users_low = fav_users[['user_id']].copy()
    users_low[[f'user_svd_{i}' for i in range(k)]] = Z_users

    out = (df_pairs
           .merge(users_low, on='user_id', how='left')
           .merge(books_low, on='book_id', how='left'))

    return out

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

    data = data.merge(books[['book_id', 'title' ,'author_id', 'author_name', 'publication_year', 'language', 'publisher', 'avg_rating', 'books_count']], on='book_id', how='left')
    # data = data.drop('book_id', axis=1)   
    print(data.head())

    ## Adding genre cosine 
    print('Calculating genre cosine...')
    fav_genre = pd.read_parquet(LOCAL_DATA_PATH / 'fav_genres.pq')

    genre_cos = add_cosine_similarity(data[['user_id', 'book_id']], fav_genre, books)
    print('cos')
    print(genre_cos.head())
    data = pd.merge(left=data, right=genre_cos, how='left', left_on=['user_id', 'book_id'], right_on=['user_id', 'book_id'])

    ## Adding lower dimensional genres
    lowdim = add_lowdim_components(data[['user_id', 'book_id']], fav_genre, books)
    print('lowdim')
    print(lowdim.head())
    data = pd.merge(left=data, right=lowdim, how='left', left_on=['user_id', 'book_id'], right_on=['user_id', 'book_id'])

    print(data.head())

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

    train = pd.read_parquet(LOCAL_DATA_PATH / 'train_hist.pq')
    books = pd.read_parquet(LOCAL_DATA_PATH / 'books.pq')
    ## Generating books grouped tables
    # print('Generating books tables...')
    # books = process_books()
    # books.to_parquet(LOCAL_DATA_PATH / 'books.pq', index=False)

    ### Generating user favourite genre
    fav_genres = build_temporal_fav_genre(train, books)
    fav_genres.to_parquet(LOCAL_DATA_PATH / 'fav_genres.pq', index=False)

    ## Ungrouping training data
    print('Ungrouping train data...')
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
