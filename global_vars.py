import os

MAIN_DATA_DIR = 'Data'
DIABETES_PATH = os.sep.join([MAIN_DATA_DIR, 'diabetes.arff'])
GERMAN_CREDIT_PATH = os.sep.join([MAIN_DATA_DIR, 'german_credit.arff'])

TF_LOGS_PATH = os.sep.join([os.curdir, 'tf_logs'])

MAIN_MODELS_DIR = 'models'

RANDOM_FOREST_PATH_FOR_DIABETES = os.sep.join([MAIN_MODELS_DIR, 'rfc_for_diabetes.pickle'])
RANDOM_FOREST_PATH_FOR_G_CREDITS = os.sep.join([MAIN_MODELS_DIR, 'rfc_for_g_credits.pickle'])

CHECKPOINT_DIR = './training_checkpoints'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")  # TODO

OVERWRITE_RFC_FILE = False
DATASET = 'diabetes'
GAN_MODE = 'cgan'

SEED = 42
BATCH_SIZE = 128
N_EPOCHS = 10
LATENT_NOISE_SIZE = 30
GENERATOR_LR = 0.0002
CRITIC_LR = 0.00002
CRITIC_DROPOUT = 0.2
CRITIC_STEPS = 3
IS_LABEL_CONDITIONAL = True
