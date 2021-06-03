from tensorflow.python.ops.numpy_ops import np_config

from GAN_model import GAN
from GAN_network_vanila import *
from random_forest_model import *

np_config.enable_numpy_behavior()  # TODO check

def section1():
    # diabetes dataset
    x_train, x_test, y_train, y_test = read_and_prepare_dataset(path_to_arff_file=DIABETES_PATH,
                                                                labels_to_num_dict={'tested_positive': 1,
                                                                                    'tested_negative': -1},
                                                                decode_categorical_columns=True)

    ds = convert_x_y_to_tf_dataset(x_train, y_train, BATCH_SIZE)
    input_size = x_train.shape[1]
    gan_model = GAN(input_size=input_size)


    gan_model.train_gan(ds, BATCH_SIZE, N_EPOCHS)

    print(gan_model.generate_samples(3))
    # german_credit dataset
    # x_train, x_test, y_train, y_test = read_and_prepare_dataset(path_to_arff_file=GERMAN_CREDIT_PATH,
    #                                                             labels_to_num_dict={'2': 1,
    #                                                                                 '1': -1},
    #                                                             decode_categorical_columns=True)



def section2():
    create_and_train_random_forest(DIABETES_PATH)
    create_and_train_random_forest(GERMAN_CREDIT_PATH)


if __name__ == '__main__':
    section1()
    # input_size = list(ds.take(1).as_numpy_iterator())[0].shape[0]
    # generator = Generator(input_size=input_size)
    # discriminator = Discriminator(input_size=input_size)
    # generated = generator.generate_sample()
    # random_real_sample = tf.convert_to_tensor(list(ds.take(1).as_numpy_iterator())[0])
    # fake_decision, real_decision = discriminator.test_generator_output(generator_sample=generated.reshape(1, -1),
    #                                                                    real_sample=random_real_sample.reshape(1, -1))

