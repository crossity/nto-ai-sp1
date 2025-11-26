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

    print(train)

if __name__ == '__main__':
    load_train()