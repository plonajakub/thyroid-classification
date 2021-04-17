import pandas as pd
import numpy as np


def get_results():
    results_df = pd.read_pickle(filepath_or_buffer='../results/results_df.pickle')
    results_np = np.zeros((results_df.shape[0], len(results_df['scores_raw'][0])))
    for idx, row in results_df.iterrows():
        results_np[idx, :] = np.array(row['scores_raw'])
    print(results_np)


def main():
    get_results()


if __name__ == '__main__':
    main()
