import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2

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


def rate_features_anova(df):
    df_values = df.values
    features = df_values[:, 0:-1]
    labels = df_values[:, -1]
    select_k_best = SelectKBest(score_func=f_classif, k='all')
    result = select_k_best.fit(features, labels)
    return result.scores_


def rate_features_mutual_info(df, discrete_features_indexes):
    df_values = df.values
    features = df_values[:, 0:-1]
    labels = df_values[:, -1]
    score_func = partial(mutual_info_classif, discrete_features=discrete_features_indexes)
    select_k_best = SelectKBest(score_func=score_func, k='all')
    result = select_k_best.fit(features, labels)
    return result.scores_


def rate_features_chi2(df):
    df_values = df.values
    features = df_values[:, 0:-1]
    labels = df_values[:, -1]
    select_k_best = SelectKBest(score_func=chi2, k='all')
    result = select_k_best.fit(features, labels)
    return result.scores_


def print_feature_scores(features, scores, title, xlabel):
    df = pd.DataFrame({'features': features, 'scores': scores})

    sorted_df = df.sort_values(by='scores')
    y_range = range(1, len(df.index) + 1)

    plt.figure(figsize=(8, 6))
    plt.hlines(y=y_range, xmin=0, xmax=sorted_df['scores'], color='skyblue')
    plt.plot(sorted_df['scores'], y_range, "o")
    plt.grid(True)
    for (_, row), y in zip(sorted_df.iterrows(), y_range):
        plt.annotate('%.2g' % row['scores'], (row['scores'] + max(scores) / 50, y - 0.15))

    plt.yticks(y_range, sorted_df['features'])
    plt.title(title, loc='left')
    plt.xlabel(xlabel)
    plt.ylabel('Feature')

    plt.tight_layout()
    plt.show()


def main():
    train_df, test_df = load_data(verbose=True)
    all_df = pd.concat([train_df, test_df])

    # ANOVA all
    scores_all = rate_features_anova(all_df)
    print_feature_scores(features_int2name, scores_all, title="ANOVA", xlabel='F')

    # ANOVA only continuous
    all_df_continuous = all_df.iloc[:, np.append(features_continuous_indexes, -1)]
    labels_continuous = [features_int2name[i] for i in features_continuous_indexes]
    scores_continuous = rate_features_anova(all_df_continuous)
    print_feature_scores(labels_continuous, scores_continuous, title="ANOVA", xlabel='F')

    # Mutual information (MI) all
    scores_mi_all = rate_features_mutual_info(all_df, discrete_features_indexes=features_categorical_indexes)
    print_feature_scores(features_int2name, scores_mi_all, title="Mutual information", xlabel='mi')

    # chi2 only categorical
    all_df_categorical = all_df.iloc[:, np.append(features_categorical_indexes, -1)]
    labels_categorical = [features_int2name[i] for i in features_categorical_indexes]
    scores_categorical = rate_features_chi2(all_df_categorical)
    print_feature_scores(labels_categorical, scores_categorical, title="Chi2", xlabel='chi2')


if __name__ == '__main__':
    main()
