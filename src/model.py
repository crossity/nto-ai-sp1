import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt

from config import *

def model_fit(train: pd.DataFrame) -> CatBoostRegressor:
    train = train[train['has_read'] == 1]
    train = train.drop('has_read', axis=1)

    train = train.sort_values(by=['timestamp'])
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

    cat_features = ['user_id', 'book_id', 'gender', 'author_id', 'language', 'publisher', 'genre_name']
    print(train.loc[:, ~train.columns.str.contains('emb')].columns)
    # print(train.iloc[:, 8])

    train_eval_split = 0.8

    num_of_rows = int(len(train) * train_eval_split)
    train_data = train.iloc[:num_of_rows]
    eval_data = train.iloc[num_of_rows:]

    train_data = Pool(data=train_data.drop(target_feature, axis=1), label=train_data[target_feature], cat_features=cat_features)
    eval_data = Pool(data=eval_data.drop(target_feature, axis=1), label=eval_data[target_feature], cat_features=cat_features)
    # X = train.drop(target_feature, axis=1)
    # y = train[target_feature]

    model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, random_state=67, verbose=50, early_stopping_rounds=20, loss_function='RMSE')
    model.fit(train_data, eval_set=eval_data)

    return model

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
    predictions = model.predict(test)

    test = test[['user_id', 'book_id']]
    test['rating'] = predictions

    return test

if __name__ == '__main__':
    train = pd.read_parquet(LOCAL_DATA_PATH / 'train.pq')
    model = model_fit(train)
    show_learning_curve(model)
    test = pd.read_parquet(LOCAL_DATA_PATH / 'test.pq')
    submit = predict(test, model)

    OUTPUT_SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    submit.to_csv(OUTPUT_SUBMISSIONS / 'submission.csv', index=False)