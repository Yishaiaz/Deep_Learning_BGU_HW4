from scipy.io import arff
import pandas as pd
from global_vars import *
from typing import List, Tuple


def read_arff_as_dataframe(path_to_arff: str) -> Tuple[pd.DataFrame, dict]:
    data = arff.loadarff(path_to_arff)
    df = pd.DataFrame(data[0])
    df_attributes = data[1]._attributes # not sure we need this
    return df, df_attributes
