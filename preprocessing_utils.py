from enum import Enum
from typing import Tuple, List
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from tensorflow.python.data import Dataset

from global_vars import *


class FeatureType(Enum):
    NUMERIC = 1
    CATEGORICAL = 2

    def is_numeric(self):
        return self.value == 1


def read_arff_file_as_dataframe(path_to_arff_file: str) -> pd.DataFrame:
    """Read and convert the given file in arff format to DataFrame"""
    data = arff.loadarff(path_to_arff_file)
    df = pd.DataFrame(data[0])

    return df


def transform_categorical_binary_column(df: pd.DataFrame, column_name: str, labels_to_num_dict: dict = None) -> pd.DataFrame:
    # df.loc[:, column_name] = df[column_name].apply(lambda x: labels_to_num_dict[x])
    df = df.apply(lambda col: col.apply(lambda row: labels_to_num_dict[row]) if col.name == column_name else col)
    return df


def gather_numeric_and_categorical_columns(df: pd.DataFrame) -> Tuple[np.array, np.array]:
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.values
    numeric_columns = df.select_dtypes(include=['number']).columns.values

    return numeric_columns, categorical_columns


def find_all_binary_columns(df: pd.DataFrame, dropna=True) -> list:
    return df.loc[:, df.nunique(dropna=dropna) == 2].columns.to_list()


def convert_x_y_to_tf_dataset(X: pd.DataFrame, y: pd.DataFrame, batch_size: int, include_y: bool = False) -> Dataset:
    ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(X.to_numpy()))

    if include_y:
        label_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y.to_numpy()))
        ds = tf.data.Dataset.zip((ds, label_ds))

    ds = ds.cache().shuffle(buffer_size=1000).batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def split_into_train_test(X: pd.DataFrame,
                          y: pd.DataFrame,
                          test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # split into train/test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED,
                                                        shuffle=True, stratify=y)

    return x_train, x_test, y_train, y_test


def encode_categorical_vars(df: pd.DataFrame, col_to_encode: str, column_to_ohe: dict) -> Tuple[pd.DataFrame, OneHotEncoder]:
    def flatten_feature_arr(features_arrays: list):
        flatten_arr = []
        for inner_array in features_arrays:
            flatten_arr += inner_array.tolist()
        return flatten_arr
    ohe = column_to_ohe[col_to_encode]
    feature_arr = ohe.fit_transform(df[[col_to_encode]]).toarray()
    features_labels = ohe.categories_
    dropped_df = df.drop(columns=[col_to_encode], inplace=False)
    encoded_df = pd.DataFrame(feature_arr, columns=flatten_feature_arr(features_labels))
    encoded_df = pd.concat([dropped_df, encoded_df], axis=1)
    return encoded_df, ohe


def read_and_prepare_dataset(path_to_arff_file: str,
                             labels_to_num_dict: dict,
                             decode_categorical_columns: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # load given file into DataFrame
    df = read_arff_file_as_dataframe(path_to_arff_file)

    target_column_name = df.columns[-1]

    # Split into X & y convention
    X, y = df.iloc[:, :-1], df.iloc[:, -1:]

    # Gather numeric and categorical columns into a list
    numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(X)

    # iterate over all categorical columns and convert decode to string
    if decode_categorical_columns:
        y = y.apply(lambda col: col.str.decode(encoding='UTF-8'))
        X = X.apply(lambda col: col.str.decode(encoding='UTF-8') if col.name in categorical_columns else col)

    # transform label categorical binary column
    y = transform_categorical_binary_column(y, target_column_name, labels_to_num_dict)

    # apply min-max scaler on all numeric columns
    column_to_scaler = {column: MinMaxScaler(feature_range=(-1, 1)) for column in numeric_columns}

    for col in numeric_columns:
        X.loc[:, col] = column_to_scaler[col].fit_transform(X[[col]])

    # # find all categorical binary columns
    # categorical_binary_columns = find_all_binary_columns(X[categorical_columns])
    #
    # # transform categorical binary columns to 0/1
    # le = LabelEncoder()
    # X = X.apply(lambda col: le.fit_transform(col) if col.name in categorical_binary_columns else col)
    #
    # # remove binary categorical columns from the general categorical column list and get new numeric columns
    # numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(X)

    # one-hot-encoding
    column_to_ohe = {column: OneHotEncoder(categories='auto') for column in categorical_columns}
    for col in categorical_columns:
        X, ohe = encode_categorical_vars(X, col_to_encode=col, column_to_ohe=column_to_ohe)

    # maps column idx to ohe and scaler
    column_idx_to_ohe = {}
    prev_column_categories = 0
    for col in categorical_columns:
        ohe = column_to_ohe[col]
        column_idx = len(numeric_columns) + prev_column_categories
        column_idx_to_ohe[column_idx] = ohe
        prev_column_categories = prev_column_categories + len(ohe.categories_[0])

    column_idx_to_scaler = {X.columns.get_loc(col): column_to_scaler[col] for col in numeric_columns}

    return X, y, column_idx_to_scaler, column_idx_to_ohe
