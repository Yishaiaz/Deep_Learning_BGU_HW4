from preprocessing_utils import *
from global_vars import *
from random_forest_model import *
from GAN_network_vanila import *

if __name__ == '__main__':
    create_and_train_random_forest(DIABETES_PATH)
    create_and_train_random_forest(G_CREDIT_PATH)
