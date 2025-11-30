from pathlib import Path

# Base data path
ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / 'data'

### INPUT DATA PATHS ###
INPUT_DATA_PATH = DATA_PATH / 'input'

BOOK_DESCRIPTIONS_PATH = INPUT_DATA_PATH / 'book_descriptions.csv'
BOOK_GENRES_PATH = INPUT_DATA_PATH / 'book_genres.csv'
BOOKS_PATH = INPUT_DATA_PATH / 'books.csv'
GENRES_PATH = INPUT_DATA_PATH / 'genres.csv'
SAMPLE_SUBMISSION_PATH = INPUT_DATA_PATH / 'sample_submission.csv'
TEST_PATH = INPUT_DATA_PATH / 'test.csv'
TRAIN_PATH = INPUT_DATA_PATH / 'train.csv'
USERS_PATH = INPUT_DATA_PATH / 'users.csv'

### OUTPUT DATA PATHS ###
OUTPUT_DATA_PATH = DATA_PATH / 'output'

OUTPUT_MODELS = OUTPUT_DATA_PATH / 'models'
OUTPUT_SUBMISSIONS = OUTPUT_DATA_PATH / 'submissions'

### LOCAL DATA PATHS ###
LOCAL_DATA_PATH = DATA_PATH / 'local'

DESCRIPTIONS_EMBEDDINGS_PATH = LOCAL_DATA_PATH / 'descriptions.pq'

### BERT ###
BERT_MAX_LENGTH = 512
BERT_BATCH_SIZE = 32 
BERT_SEED = 123
BERT_MODEL_NAME = 'DeepPavlov/rubert-base-cased'
BERT_HIDDEN_SIZE = 768
BERT_DIM = 128

### TF IDF ###
TOKEN_PATTERN_HYPHEN = r"[A-Za-zА-Яа-яЁё0-9]+(?:-[A-Za-zА-Яа-яЁё0-9]+)*"
TFIDF_DIM = 64

### MODEL ###
CAT_FEATURES = ['user_id', 'book_id', 'gender', 'author_id', 'language', 'publisher']