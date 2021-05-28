from typing import Any, List

import numpy as np
import pandas as pd

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTENC, SMOTE
from imblearn.pipeline import make_pipeline
from sklearn import clone

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate

from feature_selection import rate_features_mutual_info
from data_services import get_data
from parameters import Parameters
from constants import features_categorical_indexes, features_int2name


def create_ranking(verbose=False, random_state=None):
    all_df = get_data()
    feature_scores = rate_features_mutual_info(all_df, discrete_features_indexes=features_categorical_indexes,
                                               random_state=random_state)
    df = pd.DataFrame({'features': features_int2name, 'scores': feature_scores})
    sorted_df = df.sort_values(by='scores', ascending=False)
    if verbose:
        print('####### Feature ranking #######')
        print(sorted_df)
    return sorted_df


def train_and_test(params: Parameters, sorted_ranking_df):
    all_df = get_data()
    X = all_df.values[:, 0:-1]
    y = all_df.values[:, -1]

    sorted_features = sorted_ranking_df[:params.n_features]
    sorted_features_idx_list = sorted_features.index.to_list()
    X = X[:, sorted_features_idx_list]

    clf = MLPClassifier(solver=params.solver, max_iter=params.max_iter, learning_rate=params.learning_rate,
                        nesterovs_momentum=params.nesterovs_momentum, learning_rate_init=params.learning_rate_init,
                        hidden_layer_sizes=params.hidden_layer_sizes, momentum=params.momentum)

    if params.resample:
        current_categorical_indexes = []
        for i, feature_idx in enumerate(sorted_features_idx_list):
            if feature_idx in features_categorical_indexes:
                current_categorical_indexes.append(i)

        if len(current_categorical_indexes) == 0:
            smote = SMOTE(random_state=params.random_state, n_jobs=params.n_jobs)
        else:
            smote = SMOTENC(categorical_features=np.array(current_categorical_indexes),
                            random_state=params.random_state, n_jobs=params.n_jobs)
        smote_tomek = SMOTETomek(smote=smote, random_state=params.random_state, n_jobs=params.n_jobs)
        model = make_pipeline(smote_tomek, clf)
    else:
        model = clf

    scores = np.zeros(params.cv * params.n_experiments)
    for i in range(params.n_experiments):
        cv_res_m = cross_validate(
            clone(model), X, y, cv=params.cv, scoring=params.scoring, n_jobs=params.n_jobs, verbose=params.verbose)
        scores[i * params.cv: (i + 1) * params.cv] = cv_res_m['test_score']

    return scores


def test_param(name: str, values: List[Any], sorted_ranking_df):
    print("Testing %s..." % name)
    params = Parameters()
    results = pd.DataFrame()
    for i, v in enumerate(values):
        print("Tested value: %s; %d/%d" % (str(v), i + 1, len(values)))
        setattr(params, name, v)
        scores = train_and_test(params, sorted_ranking_df=sorted_ranking_df)
        results = results.append(
            {'param_value': v, 'scores_avg': np.mean(scores), 'scores_raw': scores},
            ignore_index=True
        )
    print("Testing of %s has been completed!" % name)
    results.to_pickle(path='../results/results_df_%s.pickle' % name)
    results.to_excel('../results/%s.xlsx' % name)
    results.to_csv(path_or_buf='../results/%s.csv' % name)
    print("Results has been written!")


def main():
    sorted_ranking_df = create_ranking(verbose=True, random_state=Parameters().random_state)

    test_param('resample', [False, True], sorted_ranking_df=sorted_ranking_df)
    test_param('n_features', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], sorted_ranking_df=sorted_ranking_df)
    test_param('hidden_layer_sizes', [(5,), (25,), (125,), (625,), (3125,)], sorted_ranking_df=sorted_ranking_df)
    test_param('learning_rate_init', [0.01, 0.1, 0.3, 0.6, 0.9, 1], sorted_ranking_df=sorted_ranking_df)
    test_param('learning_rate', ['constant', 'invscaling', 'adaptive'], sorted_ranking_df=sorted_ranking_df)
    test_param('momentum', [0, 0.2, 0.4, 0.6, 0.8, 1], sorted_ranking_df=sorted_ranking_df)


if __name__ == '__main__':
    main()
