import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from tabulate import tabulate
from constants import clfs_names_column, clfs_names_row

alpha = .05


def get_results():
    results_df = pd.read_pickle(
        filepath_or_buffer='../results/results_df.pickle')
    results_np = np.zeros(
        (results_df.shape[0], len(results_df['scores_raw'][0])))
    for idx, row in results_df.iterrows():
        results_np[idx, :] = np.array(row['scores_raw'])
    results_np_table = np.concatenate((clfs_names_column, results_np), axis=1)
    results_np_table = tabulate(results_np_table, floatfmt=".3f")
    print('Number of used features (from ranking): %d' % results_df['n_features'][0])
    print('\n##################### classifiers\' scores ######################')
    print(results_np_table)
    return results_np


def analyze(results):
    t_statistic = np.zeros((len(results), len(results)))
    p_value = np.zeros((len(results), len(results)))

    # t-statistic and p value matrixes
    for i in range(len(results)):
        for j in range(len(results)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(
                results[i], results[j])

    t_statistic_table = np.concatenate(
        (clfs_names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(
        t_statistic_table, clfs_names_row, floatfmt=".2f")

    p_value_table = np.concatenate((clfs_names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, clfs_names_row, floatfmt=".2f")

    # advantage matrix
    advantage = np.zeros((len(results), len(results)))
    advantage[t_statistic > 0] = 1
    advantage_table = np.concatenate((clfs_names_column, advantage), axis=1)
    advantage_table = tabulate(advantage_table, clfs_names_row)

    # statistical significance matrix
    significance = np.zeros((len(results), len(results)))
    significance[p_value <= alpha] = 1
    significance_table = np.concatenate(
        (clfs_names_column, significance), axis=1)
    significance_table = tabulate(significance_table, clfs_names_row)

    # statistically significantly better
    statistically_better = significance*advantage
    statistically_better_table = np.concatenate(
        (clfs_names_column, statistically_better), axis=1)
    statistically_better_table = tabulate(
        statistically_better_table, clfs_names_row)
    return t_statistic_table, p_value_table, advantage_table, significance_table, statistically_better_table


def main():
    results = get_results()
    t_statistic, p_value, advantage, significance, statistically_better = analyze(
        results)
    print('\n######################### t-statistic ###########################')
    print(t_statistic)
    print('\n########################## p values #############################')
    print(p_value)
    print('\n####################### advantage matrix ########################')
    print(advantage)
    print('\n################## statistical significance #####################')
    print(significance)
    print('\n############# statistically significantly better ################')
    print(statistically_better)


if __name__ == '__main__':
    main()
