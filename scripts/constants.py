import numpy as np

features_name2int = {
    'age': 0,
    'sex': 1,
    'on_thyroxine': 2,
    'query_on_thyroxine': 3,
    'on_antithyroid_medication': 4,
    'sick': 5,
    'pregnant': 6,
    'thyroid_surgery': 7,
    'i131_treatment': 8,
    'query_hypothyroid': 9,
    'query_hyperthyroid': 10,
    'lithium': 11,
    'goiter': 12,
    'tumor': 13,
    'hypopituitary': 14,
    'psych': 15,
    'tsh': 16,
    't3': 17,
    'tt4': 18,
    't4u': 19,
    'fti': 20
}
features_int2name = np.array(list(features_name2int.keys()))
data_column_names = np.append(features_int2name, 'class')

features_continuous_indexes = np.array([0, 16, 17, 18, 19, 20])
features_categorical_indexes = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])


classes_name2int = {
    'normal_condition': 1,
    'hyperthyroidism': 2,
    'hypothyroidism': 3
}
classes_listed = np.array(list(classes_name2int.keys()))
classes_int2name = np.array([None] + list(classes_name2int.keys()))

clfs_names_row = ["hl_5_with_momentum", "hl_5_no_momentum",
                  "hl_25_with_momentum", "hl_25_no_momentum", "hl_125_with_momentum", "hl_125_no_momentum"]

clfs_names_column = np.array([["hl_5_with_momentum"], ["hl_5_no_momentum"],
                              ["hl_25_with_momentum"], ["hl_25_no_momentum"], ["hl_125_with_momentum"], ["hl_125_no_momentum"]])
