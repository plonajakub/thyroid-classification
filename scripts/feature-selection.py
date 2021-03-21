import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from constants import *


def load_data(verbose=False):
    train_df = pd.read_csv('../data/ann-train.data', sep=' ', header=None)
    train_df = train_df.drop(train_df.columns[[22, 23]], axis=1)
    train_df.columns = data_column_names

    pd.set_option('display.max_columns', None)
    if verbose:
        print('###### ann-train.data - data ######')
        print(train_df)
        print('###### ann-train.data - types ######')
        print(train_df.dtypes)
        print('###### ann-train.data - statistics ######')
        print(train_df.describe())

    test_df = pd.read_csv('../data/ann-test.data', sep=' ', header=None)
    test_df = test_df.drop(test_df.columns[[22, 23]], axis=1)
    test_df.columns = data_column_names

    if verbose:
        print('###### ann-test.data - data ######')
        print(test_df)
        print('###### ann-test.data - types ######')
        print(test_df.dtypes)
        print('###### ann-test.data - statistics ######')
        print(test_df.describe())
    pd.set_option('display.max_columns', 0)

    return train_df, test_df


def rate_features(df):
    df_values = df.values
    features = df_values[:, 0:-1]
    labels = df_values[:, -1]
    select_k_best = SelectKBest(score_func=f_classif, k='all')
    result = select_k_best.fit(features, labels)
    return result.scores_


def print_feature_scores(scores):
    df = pd.DataFrame({'features': features_listed, 'scores': scores})

    sorted_df = df.sort_values(by='scores')
    y_range = range(1, len(df.index) + 1)

    plt.figure(figsize=(8, 6))
    plt.hlines(y=y_range, xmin=0, xmax=sorted_df['scores'], color='skyblue')
    plt.plot(sorted_df['scores'], y_range, "o")
    plt.grid(True)
    for (_, row), y in zip(sorted_df.iterrows(), y_range):
        plt.annotate(round(row['scores'], 2), (row['scores'] + 100, y - 0.25))

    plt.yticks(y_range, sorted_df['features'])
    plt.title("ANOVA F-values", loc='left')
    plt.xlabel('F-value')
    plt.ylabel('Feature')

    plt.tight_layout()
    plt.show()


def main():
    train_df, test_df = load_data(verbose=True)
    all_df = pd.concat([train_df, test_df])
    scores = rate_features(all_df)
    print_feature_scores(scores)


if __name__ == '__main__':
    main()
