import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt

from src.config import *

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

    eval_last = eval_last.drop('title', axis=1)
    eval_last = eval_last.drop('timestamp', axis=1)

    train_hist = train_hist.drop('title', axis=1)
    train_hist = train_hist.drop('timestamp', axis=1)
    return train_hist, eval_last

def model_fit(train: pd.DataFrame) -> CatBoostRegressor:
    train = train[train['has_read'] == 1]
    train = train.drop('has_read', axis=1)

    # train = train.sort_values(by=['timestamp'])
    train_data, eval_data = split_last_item_per_user(train)
    train = train.drop('timestamp', axis=1)

    train = train.drop('title', axis=1)

    # cat_features = train.loc[:, ~train.columns.str.contains('emb')].columns.to_numpy().tolist()

    target_feature = 'rating'
    # cat_features.remove(target_feature)

    # num_features = [
    #     'age', 'publication_year', 'avg_rating', 'books_count'
    # ]

    # for f in num_features:
    #     cat_features.remove(f)
    # print(train.loc[:, ~train.columns.str.contains('emb')].columns)
    # print(train.iloc[:, 8])

    # train_eval_split = 0.98

    # num_of_rows = int(len(train) * train_eval_split)
    # train_data = train.iloc[:num_of_rows]
    # eval_data = train.iloc[num_of_rows:]\

    ## Selecting last book rated by every user

    train_data = Pool(data=train_data.drop(target_feature, axis=1), label=train_data[target_feature], cat_features=CAT_FEATURES)
    eval_data = Pool(data=eval_data.drop(target_feature, axis=1), label=eval_data[target_feature], cat_features=CAT_FEATURES)
    # X = train.drop(target_feature, axis=1)
    # y = train[target_feature]

    model = CatBoostRegressor(iterations=4000, learning_rate=0.03, depth=8, l2_leaf_reg=15, subsample=0.8, rsm=0.8, random_state=67, loss_function='RMSE', early_stopping_rounds=200)
    model.fit(train_data, eval_set=eval_data)

    return model

def model_save(model: CatBoostRegressor):
    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)
    model.save_model(OUTPUT_MODELS / 'model.cbm',
            format="cbm")
    
def model_load() -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(OUTPUT_MODELS / 'model.cbm')

def show_learning_curve(model: CatBoostRegressor):
    evals_result = model.get_evals_result()
    train_loss = evals_result['learn']['RMSE']
    eval_loss = evals_result['validation']['RMSE']

    iterations = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, train_loss, label='Training Loss', color='blue')
    plt.plot(iterations, eval_loss, label='Validation Loss', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('CatBoost Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

def predict(test: pd.DataFrame, model: CatBoostRegressor) -> pd.DataFrame:
    print(test.loc[:, ~test.columns.str.contains('emb')].columns)
    test = test.drop('title', axis=1)

    test_pool = Pool(data=test, cat_features=CAT_FEATURES)
    predictions = model.predict(test_pool)

    test = test[['user_id', 'book_id']]
    test['rating_predict'] = predictions

    return test

if __name__ == '__main__':
    train = pd.read_parquet(LOCAL_DATA_PATH / 'train.pq')
    model = model_fit(train)
    model_save(model)

    # show_learning_curve(model)

    test = pd.read_parquet(LOCAL_DATA_PATH / 'test.pq')
    submit = predict(test, model)

    OUTPUT_SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    submit.to_csv(OUTPUT_SUBMISSIONS / 'submission.csv', index=False)