import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, cross_validate

from feature_selection import rate_features_mutual_info
from data_services import get_data
from parameters import Parameters
from constants import features_categorical_indexes, features_int2name
from misc import avg


def train_and_test(params, debug=False):
    all_df = get_data()
    X = all_df.values[:, 0:-1]
    y = all_df.values[:, -1]

    # feature ranking
    feature_scores = rate_features_mutual_info(
        all_df, discrete_features_indexes=features_categorical_indexes)
    df = pd.DataFrame(
        {'features': features_int2name, 'scores': feature_scores})
    if debug:
        print(df)

    # sorting features and preparing data
    sorted_df = df.sort_values(by='scores', ascending=False)
    print('### Feature ranking ###')
    print(sorted_df)

    sorted_features = sorted_df[:params.n_features_limit]
    if debug:
        print(sorted_features)
    sorted_features_idx_list = sorted_features.index.to_list()
    X = X[:, sorted_features_idx_list]
    if debug:
        print(X)

    clf = MLPClassifier(solver='sgd', max_iter=params.max_iter, learning_rate=params.learning_rate,
                        nesterovs_momentum=False, learning_rate_init=params.learning_rate_init)

    # experiments
    results = pd.DataFrame(
        columns=['n_features', 'hidden_layer_size', 'with_momentum', 'score'])
    i_ext = 1
    i_ext_max = params.n_features_limit * len(params.hidden_layer_sizes)
    for n_features in range(params.n_features_limit):
        X_data = X[:, range(n_features + 1)]
        if debug:
            print(X_data)
        for n in params.hidden_layer_sizes:
            clf.hidden_layer_sizes = (n,)
            scores = {'with_momentum': [], 'no_momentum': []}
            i_int = [1, 1]
            for i in range(params.n_experiments):
                # sgd with momentum
                print_iteration_info(
                    i_ext, i_ext_max, n_features + 1, n, i_int[0], params.n_experiments, 'Y')
                clf.momentum = params.momentum
                score = cross_validate(
                    clf, X_data, y, cv=2, scoring='accuracy', n_jobs=-1, verbose=3, return_estimator=True)
                scores['with_momentum'].append(avg(score['test_score']))
                if debug:
                    print(score)
                i_int[0] += 1

                # sgd without momentum
                print_iteration_info(
                    i_ext, i_ext_max, n_features + 1, n, i_int[1], params.n_experiments, 'N')
                clf.momentum = 0
                score = cross_validate(
                    clf, X_data, y, cv=2, scoring='accuracy', n_jobs=-1, verbose=3, return_estimator=True)
                scores['no_momentum'].append(avg(score['test_score']))
                if debug:
                    print(score)
                i_int[1] += 1
            results = results.append({'n_features': n_features + 1, 'hidden_layer_size': n, 'with_momentum': True,
                                      'score': avg(scores['with_momentum'])}, ignore_index=True)
            results = results.append({'n_features': n_features + 1, 'hidden_layer_size': n, 'with_momentum': False,
                                      'score': avg(scores['no_momentum'])}, ignore_index=True)
            i_ext += 1
    return results


def print_iteration_info(i, i_max, n_features, hl_size, i_exp, i_exp_max, with_momentum):
    info_string = '### iteration: %d/%d # n_features: %d # hl_size: %d # experiment: %d/%d # with_momentum: %s ###' % (
        i, i_max, n_features, hl_size, i_exp, i_exp_max, with_momentum)
    print(info_string)


def main():
    base_params = Parameters()
    results = train_and_test(base_params)
    print('############# Final results ################')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(results)


if __name__ == '__main__':
    main()
