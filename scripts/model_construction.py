import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from feature_selection import load_data, rate_features_mutual_info
from constants import *


def main():
    train_df, test_df = load_data(verbose=False)
    all_df = pd.concat([train_df, test_df])
    X = all_df.values[:, 0:-1]
    y = all_df.values[:, -1]

    # scalling data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # feature ranking
    feature_scores = rate_features_mutual_info(
        all_df, discrete_features_indexes=features_categorical_indexes)

    df = pd.DataFrame(
        {'features': features_int2name, 'scores': feature_scores})
    print(features_int2name)

    # sorting features and preparing data
    sorted_df = df.sort_values(by='scores', ascending=False)
    print(sorted_df)
    sorted_features = sorted_df.loc[sorted_df.index[0:7], 'features']
    print(sorted_features)

    X = X[:, [16, 20, 17, 18, 19, 2, 0]]
    print(X)

    # best parameters search
    mlp = MLPClassifier(solver='sgd', max_iter=100,
                        nesterovs_momentum=False, random_state=1)

    parameters = {'alpha': [0.00001, 0.0001, 0.05, 0.1, 0.2, 0.5, 1],
                  'learning_rate': ['constant', 'adaptive']}

    clf = GridSearchCV(mlp, parameters, cv=2, scoring="accuracy")
    clf.fit(X, y)
    print("Best parameters found:\n", clf.best_params_)

    # experiments
    for j in range(0, 7):
        X_data = X[:, range(0, j+1)]
        print("Features count:" + str(j+1))
        print(X_data)
        for n in range(5, 16, 5):
            print("Results for MLP with "+str(n)+" neurons in hidden layer")
            clf.hidden_layer_sizes = (n,)
            for i in range(2):
                # sgd with momentum
                print("With momentum")
                clf.momentum = 0.9

                score = cross_val_score(
                    clf, X_data, y, cv=2, scoring='accuracy', n_jobs=-1)
                print(score)

                # sgd without momentum
                print("without momentum")
                clf.momentum = 0

                score = cross_val_score(
                    clf, X_data, y, cv=2, scoring='accuracy', n_jobs=-1)
                print(score)


if __name__ == '__main__':
    main()
