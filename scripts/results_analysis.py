from copy import copy

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from tabulate import tabulate


def print_tabulate(matrix, axis_names, title):
    matrix_tabulated = np.concatenate((np.expand_dims(axis_names, 1), matrix), axis=1)
    matrix_tabulated = tabulate(matrix_tabulated, axis_names)
    print('######################### %s ###########################' % title)
    print(matrix_tabulated)
    print()


def get_results(path_to_pickle: str, param_name: str):
    results_df = pd.read_pickle(filepath_or_buffer=path_to_pickle)
    results_np = np.zeros((results_df.shape[0], len(results_df['scores_raw'][0])))
    matrix_names = []
    for idx, row in results_df.iterrows():
        results_np[idx, :] = np.array(row['scores_raw'])
        matrix_names.append(param_name + '_' + str(row['param_value']))
    return results_np, matrix_names


def analyze(results, names, param_title, alpha=0.05):
    t_statistic = np.zeros((len(results), len(results)))
    p_value = np.zeros((len(results), len(results)))

    # t-statistic and p value matrices
    for i in range(len(results)):
        for j in range(len(results)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(
                results[i], results[j])

    print_tabulate(t_statistic, names, '%s - t_statistic' % param_title)
    print_tabulate(p_value, names, '%s - p_value' % param_title)

    # advantage matrix
    advantage = np.zeros((len(results), len(results)))
    advantage[t_statistic > 0] = 1
    print_tabulate(advantage, names, '%s - advantage' % param_title)

    # statistical significance matrix
    significance = np.zeros((len(results), len(results)))
    significance[p_value <= alpha] = 1
    print_tabulate(significance, names, '%s - significance' % param_title)

    # statistically significantly better
    statistically_better = significance * advantage
    print_tabulate(statistically_better, names, '%s - statistically better' % param_title)


def main():
    analyze(*get_results('../results/results_df_resample.pickle', 'resample'), 'Resample')
    analyze(*get_results('../results/results_df_n_features.pickle', 'n_features'), 'Number of features')
    analyze(*get_results('../results/results_df_hidden_layer_sizes.pickle', 'hidden_layer_sizes'), 'Number of neurons')
    analyze(*get_results('../results/results_df_learning_rate_init.pickle', 'learning_rate_init'),
            'Learning rate initial value')
    analyze(*get_results('../results/results_df_learning_rate.pickle', 'learning_rate'), 'Learning rate strategy')
    analyze(*get_results('../results/results_df_momentum.pickle', 'momentum'), 'Momentum')


if __name__ == '__main__':
    main()
