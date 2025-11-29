import pandas as pd

from src.config import *

df = pd.read_csv(LOCAL_DATA_PATH / 'descriptions.csv')
df.to_parquet(LOCAL_DATA_PATH / 'descriptions.pq', index=False)