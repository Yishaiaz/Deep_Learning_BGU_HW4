import os.path
import pickle
from sklearn.ensemble import RandomForestClassifier
from preprocessing_utils import *


def create_and_train_random_forest(file_path_to_train: str) -> RandomForestClassifier:
    x_train, x_test, y_train, y_test = read_and_prepare_dataset(file_path_to_train, test_size=0.33, for_rfc=True)
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    print(f"rfc score on {file_path_to_train.split(os.sep)[-1]} dataset: {rfc.score(x_test, y_test)}") # todo: remove before submission
    path_for_rfc = RANDOM_FOREST_PATH_FOR_DIABETES if \
        file_path_to_train.split(os.sep)[-1] == RANDOM_FOREST_PATH_FOR_DIABETES else RANDOM_FOREST_PATH_FOR_G_CREDITS
    if OVERWRITE_RFC_FILE or not os.path.isfile(path_for_rfc):
        with open(path_for_rfc, 'wb') as rfc_pickle_file:
            pickle.dump(rfc, rfc_pickle_file)

    return rfc


def load_trained_random_forest(rfc_type: str) -> RandomForestClassifier:
    if rfc_type == 'd':
        path_to_pickle = RANDOM_FOREST_PATH_FOR_DIABETES
    elif rfc_type == 'g':
        path_to_pickle = RANDOM_FOREST_PATH_FOR_G_CREDITS
    else:
        raise ValueError(f"rfc_type can be either 'd' for Diabetes trained rfc or 'g' for german_credits trained rfc\n"
                         f"rfc_type recieved {rfc_type}")
    if os.path.isfile(path_to_pickle):
        rfc = None
        with open(path_to_pickle, 'rb') as rfc_file:
            rfc = pickle.load(rfc_file)
        return rfc

    else:
        raise IOError(f"no file was found at {path_to_pickle}\n "
                      f"try invoking the function 'create_and_train_random_forest' first")
