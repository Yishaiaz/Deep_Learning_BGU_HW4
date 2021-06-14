import os

MAIN_DATA_DIR = 'Data'
DIABETES_PATH = os.sep.join([MAIN_DATA_DIR, 'diabetes.arff'])
GERMAN_CREDIT_PATH = os.sep.join([MAIN_DATA_DIR, 'german_credit.arff'])

TF_LOGS_PATH = os.sep.join([os.curdir, 'tf_logs'])

MAIN_MODELS_DIR = 'models'

OVERWRITE_RFC_FILE = False
DATASET = 'diabetes'
GAN_MODE = 'cgan'
SECTION = "section1"

SEED = 42
BATCH_SIZE = 64
N_EPOCHS = 1
LATENT_NOISE_SIZE = 30
GENERATOR_LR = 0.00002
CRITIC_LR = 0.00002
CRITIC_DROPOUT = 0.2
CRITIC_STEPS = 3
IS_LABEL_CONDITIONAL = True
NUM_OF_RANDOM_SAMPLES = 100
