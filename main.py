from preprocessing_utils import *
from global_vars import *
from random_forest_model import *
from GAN_network_vanila import *

if __name__ == '__main__':
    d_data_df, d_data_attributes = read_arff_as_dataframe(DIABETES_PATH)
    g_data_df, g_data_attributes = read_arff_as_dataframe(G_CREDIT_PATH)
    print()