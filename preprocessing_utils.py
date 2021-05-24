import numpy as np
import pandas as pd
from global_vars import *
from typing import List, Tuple
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict


def read_arff_as_dataframe(path_to_arff: str) -> Tuple[pd.DataFrame, dict]:
    data = arff.loadarff(path_to_arff)
    df = pd.DataFrame(data[0])
    df_attributes = data[1]._attributes # not sure we need this
    return df, df_attributes


def read_and_prepare_dataset(path_to_arff: str, test_size: float = 0.33) -> Tuple[np.array, np.array, np.array, np.array]:
    data_df, data_attributes = read_arff_as_dataframe(path_to_arff=path_to_arff)
    label_encoders_dict = defaultdict(LabelEncoder)
    # encode categorical columns
    cols_to_encode = data_df.select_dtypes(include=['object']).copy().columns
    fitted_df = data_df.apply(lambda x: label_encoders_dict[x.name].fit_transform(x) if x.name in cols_to_encode else x)
    # minmax scaling for all columns
    scaled_df = fitted_df.apply(lambda x: (x - x.values.min()) / (x.values.max() - x.values.min()))
    # split to x and y
    only_vals = scaled_df.values
    x, y = only_vals[:, :-1], only_vals[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=SEED)
    return x_train, x_test, y_train, y_test

