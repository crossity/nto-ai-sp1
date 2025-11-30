import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from src.config import *
from src.data_load import *

def __genre_name_fixer(name: str) -> str:
    return name.replace('-', ' ')


def create_genre_embeddings():
    genres = load_genres()

    vec = TfidfVectorizer(
        analyzer="word",
        lowercase=True,                 
        token_pattern=TOKEN_PATTERN_HYPHEN,
        ngram_range=(1, 1),             
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        dtype=np.float32,              
    )

    texts = genres['genre_name'].map(__genre_name_fixer).tolist()

    X = vec.fit_transform(texts)

    svd = TruncatedSVD(n_components=TFIDF_DIM, random_state=0)
    pipe = make_pipeline(svd, Normalizer(copy=False))

    X_red = pipe.fit_transform(X).astype(np.float32)

    coord_cols = {f"vec_{i}": X_red[:, i] for i in range(X_red.shape[1])}
    # out = pd.DataFrame({
    #     'genre_id': genres['genre_id'],
    #     'books_count': genres['books_count']
    # }) | pd.DataFrame(coord_cols)
    # out = pd.DataFrame()
    # out['genre_id'] = genres['genre_id']
    # out['books_count'] = 
    genres = genres.drop('genre_name', axis=1)
    # genres = genres._append(pd.DataFrame(coord_cols), ignore_index=True)
    genres = pd.concat([genres, pd.DataFrame(coord_cols)], axis=1)

    LOCAL_DATA_PATH.mkdir(parents=True, exist_ok=True)
    genres.to_parquet(LOCAL_DATA_PATH / 'genres.pq', index=False)

if __name__ == '__main__':
    create_genre_embeddings()