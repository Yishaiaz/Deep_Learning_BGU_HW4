import pickle
from sklearn.ensemble import RandomForestClassifier
from preprocessing_utils import *


def create_and_train_random_forest(file_path_to_train: str):
    data_df, df_attributes = read_arff_as_dataframe(file_path_to_train)
    data_values = data_df.values
    x, y = data_values[:, :-1], data_values[:, -1]


def load_trained_random_forest() -> RandomForestClassifier:
    pass
