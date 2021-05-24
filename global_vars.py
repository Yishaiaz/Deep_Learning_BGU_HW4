import os

MAIN_DATA_DIR = 'Data'
DIABETES_PATH = os.sep.join([MAIN_DATA_DIR, 'diabetes.arff'])
G_CREDIT_PATH = os.sep.join([MAIN_DATA_DIR, 'german_credit.arff'])

MAIN_MODELS_DIR = 'Models'

RANDOM_FOREST_PATH_FOR_DIABETES = os.sep.join([MAIN_MODELS_DIR, 'rfc_for_diabetes.pickle'])
RANDOM_FOREST_PATH_FOR_G_CREDITS = os.sep.join([MAIN_MODELS_DIR, 'rfc_for_g_credits.pickle'])
OVERWRITE_RFC_FILE = False


SEED = 42
