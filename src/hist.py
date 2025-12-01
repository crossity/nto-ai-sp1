import pandas as pd

from src.config import *
from src.data_load import *

def split_last_item_per_user(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = interactions.copy()
    if df['timestamp'].dtype.kind in {'O', 'U', 'S'}:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.sort_values(by=['timestamp'])

    last_idx = df.groupby('user_id', sort=False)['timestamp'].idxmax()

    eval_last = df.loc[last_idx].copy()
    train_hist = df.drop(index=last_idx).copy()

    eval_ts = eval_last[['user_id', 'timestamp']]
    train_hist = train_hist.merge(
        eval_ts.assign(flag=1),
        on=['user_id', 'timestamp'],
        how='left'
    )
    train_hist = train_hist[train_hist['flag'].isna()].drop(columns='flag')

    return train_hist, eval_last

if __name__ == '__main__':
    train = load_train()
    train_hist, eval_hist = split_last_item_per_user(train)

    train_hist.to_parquet(LOCAL_DATA_PATH / 'train_hist.pq', index=False)
    eval_hist.to_parquet(LOCAL_DATA_PATH / 'eval_hist.pq', index=False)