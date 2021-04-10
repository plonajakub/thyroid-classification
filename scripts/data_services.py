import pandas as pd
from sklearn.preprocessing import StandardScaler

from constants import data_column_names, features_int2name


def load_data(verbose=False):
    train_df = pd.read_csv('../data/ann-train.data', sep=' ', header=None)
    train_df = train_df.drop(train_df.columns[[22, 23]], axis=1)
    train_df.columns = data_column_names

    if verbose:
        with pd.set_option('display.max_columns', None):
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
        with pd.set_option('display.max_columns', None):
            print('###### ann-test.data - data ######')
            print(test_df)
            print('###### ann-test.data - types ######')
            print(test_df.dtypes)
            print('###### ann-test.data - statistics ######')
            print(test_df.describe())

    return train_df, test_df


def preprocess_data(df):
    df = make_unique(df)
    df = scale(df)
    df = resample(df)
    return df


def make_unique(df):
    df = df.drop_duplicates()  # regular duplicates
    df = df.drop_duplicates(subset=features_int2name, keep=False)  # conflicting data
    return df


def scale(df):
    # TODO scale only continuous values
    # X = df.values[:, 0:-1]
    # y = df.values[:, -1]
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    # df = pd.DataFrame(columns=features_int2name, data=X)
    # df['class'] = y
    return df


def resample(df):
    # TODO implement
    return df


def get_data(preprocess=True):
    train_df, test_df = load_data()
    all_df = pd.concat([train_df, test_df])
    if preprocess:
        all_df = preprocess_data(all_df)
    return all_df
