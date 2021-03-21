import numpy as np

features_name2int = {
    'age': 1,
    'sex': 2,
    'on_thyroxine': 3,
    'query_on_thyroxine': 4,
    'on_antithyroid_medication': 5,
    'sick': 6,
    'pregnant': 7,
    'thyroid_surgery': 8,
    'i131_treatment': 9,
    'query_hypothyroid': 10,
    'query_hyperthyroid': 11,
    'lithium': 12,
    'goiter': 13,
    'tumor': 14,
    'hypopituitary': 15,
    'psych': 16,
    'tsh': 17,
    't3': 18,
    'tt4': 19,
    't4u': 20,
    'fti': 21
}
features_int2name = np.array(list(features_name2int.keys()))
data_column_names = np.append(features_int2name, 'class')

classes_name2int = {
    'normal_condition': 1,
    'hyperthyroidism': 2,
    'hypothyroidism': 3
}
classes_int2name = np.array(list(classes_name2int.keys()))
