from enum import Enum
from typing import Tuple, List

import pandas as pd
import tensorflow as tf
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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
    df.loc[:, column_name] = df[column_name].apply(lambda x: labels_to_num_dict[x])

    return df


def get_feature_to_type_mapping(df: pd.DataFrame) -> dict:
    """Returns a dictionary which holds a mapping between feature<->feature-type"""
    feature_to_type = {}

    for c in df.columns:
        if is_string_dtype(df[c]):
            feature_to_type[c] = FeatureType.CATEGORICAL
        elif is_numeric_dtype(df[c]):
            feature_to_type[c] = FeatureType.NUMERIC
        else:
            raise RuntimeError("invalid column type for [{}]".format(c))

    return feature_to_type


def gather_numeric_and_categorical_columns(df: pd.DataFrame) -> Tuple[List, List]:
    numeric_columns = []
    categorical_columns = []

    for feature, feature_type in get_feature_to_type_mapping(df).items():
        if feature_type.is_numeric():
            numeric_columns.append(feature)
        else:
            categorical_columns.append(feature)

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
                          test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # split into train/test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED,
                                                        shuffle=True, stratify=y)

    return x_train, x_test, y_train, y_test


def read_and_prepare_dataset(path_to_arff_file: str,
                             labels_to_num_dict: dict,
                             decode_categorical_columns: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # load given file into DataFrame
    df = read_arff_file_as_dataframe(path_to_arff_file)

    target_column_name = df.columns[-1]

    # Split into X & y convention
    X, y = df.iloc[:, :-1], df.iloc[:, -1:]

    # Gather numeric and categorical columns into a list
    numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(X)

    # iterate over all categorical columns and convert decode to string
    if decode_categorical_columns:
        y.loc[:, target_column_name] = y[target_column_name].apply(lambda label: label.decode())
        for categorical_column in categorical_columns:
            X.loc[:, categorical_column] = X[categorical_column].apply(lambda x: x.decode())

    # transform label categorical binary column
    y = transform_categorical_binary_column(y, target_column_name, labels_to_num_dict)

    # apply min-max scaler on all numeric columns
    X[numeric_columns] = pd.DataFrame(MinMaxScaler().fit_transform(X[numeric_columns].values), columns=numeric_columns, index=X.index)

    # align categorical column type
    col_type_mapping = {col: 'object' for col in categorical_columns}
    X = X.astype(col_type_mapping)

    # find all categorical binary columns
    categorical_binary_columns = find_all_binary_columns(X[categorical_columns])

    # transform categorical binary columns to 0/1
    le = LabelEncoder()
    for categorical_binary_column in categorical_binary_columns:
        X.loc[:, categorical_binary_column] = le.fit_transform(X[categorical_binary_column])

    # remove binary categorical columns from the general categorical column list
    categorical_columns = [column for column in categorical_columns if column not in set(categorical_binary_columns)]

    # one-hot-encoding
    X = pd.get_dummies(X, prefix=categorical_columns)

    return X, y
