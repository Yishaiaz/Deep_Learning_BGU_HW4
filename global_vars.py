import os

MAIN_DATA_DIR = 'Data'
DIABETES_PATH = os.sep.join([MAIN_DATA_DIR, 'diabetes.arff'])
GERMAN_CREDIT_PATH = os.sep.join([MAIN_DATA_DIR, 'german_credit.arff'])

MAIN_MODELS_DIR = 'models'

RANDOM_FOREST_PATH_FOR_DIABETES = os.sep.join([MAIN_MODELS_DIR, 'rfc_for_diabetes.pickle'])
RANDOM_FOREST_PATH_FOR_G_CREDITS = os.sep.join([MAIN_MODELS_DIR, 'rfc_for_g_credits.pickle'])

CHECKPOINT_DIR = './training_checkpoints'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")

OVERWRITE_RFC_FILE = False


SEED = 42
BATCH_SIZE = 32
N_EPOCHS = 30
LATENT_NOISE_SIZE = 30
GENERATOR_LR = 0.0005
CRITIC_LR = 0.0005
