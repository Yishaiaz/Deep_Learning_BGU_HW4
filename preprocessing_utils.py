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


def encode_categorical_vars(df: pd.DataFrame, cols_to_encode: List) -> Tuple[pd.DataFrame, OneHotEncoder]:
    def flatten_feature_arr(features_arrays: list):
        flatten_arr = []
        for inner_array in features_arrays:
            flatten_arr += inner_array.tolist()
        return flatten_arr
    ohe = OneHotEncoder(categories='auto')
    feature_arr = ohe.fit_transform(df[cols_to_encode]).toarray()
    features_labels = ohe.categories_
    dropped_df = df.drop(columns=cols_to_encode, inplace=False)
    encoded_df = pd.DataFrame(feature_arr, columns=flatten_feature_arr(features_labels))
    encoded_df = pd.concat([encoded_df, dropped_df], axis=1)
    return encoded_df, ohe


def read_and_prepare_dataset(path_to_arff_file: str,
                             labels_to_num_dict: dict,
                             decode_categorical_columns: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
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
    X = X.apply(lambda col: MinMaxScaler(feature_range=(-1, 1)).fit_transform(np.asarray(col).reshape(-1, 1)).flatten() if col.name in numeric_columns else col)

    # find all categorical binary columns
    categorical_binary_columns = find_all_binary_columns(X[categorical_columns])

    # transform categorical binary columns to 0/1
    le = LabelEncoder()
    X = X.apply(lambda col: le.fit_transform(col) if col.name in categorical_binary_columns else col)

    # remove binary categorical columns from the general categorical column list
    categorical_columns = [column for column in categorical_columns if column not in set(categorical_binary_columns)]

    # one-hot-encoding,
    X, ohe = encode_categorical_vars(X, cols_to_encode=categorical_columns) # todo: return ohes as well for inverse transorm if necessary

    return X, y, ohe
