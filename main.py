from tensorflow.python.ops.numpy_ops import np_config

from GAN_model import GAN
from random_forest_model import *
import matplotlib.pyplot as plt

np_config.enable_numpy_behavior()  # TODO check


def plot_critic_generator_loss(x1: List, y1: List, x2: List, y2: List, label1: str, label2: str, x_axis: str, y_axis: str, title: str):
    # plotting the line 1 points
    plt.plot(x1, y1, label=label1)
    # plotting the line 2 points
    plt.plot(x2, y2, label=label2)
    plt.xlabel(x_axis)
    # Set the y axis label of the current axis.
    plt.ylabel(y_axis)
    # Set a title of the current axes.
    plt.title(title)
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()


def plot_critic_accuracy(x: List, y: List, label1: str, x_axis: str, y_axis: str, title: str):
    plt.plot(x, y, label=label1)
    plt.xlabel(x_axis)
    # Set the y axis label of the current axis.
    plt.ylabel(y_axis)
    # Set a title of the current axes.
    plt.title(title)
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()


def section1():
    # diabetes dataset
    X, y = read_and_prepare_dataset(path_to_arff_file=DIABETES_PATH,
                                    labels_to_num_dict={'tested_positive': 1,
                                                        'tested_negative': -1},
                                    decode_categorical_columns=True)
    # german credit card dataset
    # X, y = read_and_prepare_dataset(path_to_arff_file=GERMAN_CREDIT_PATH,
    #                                 labels_to_num_dict={'1': 1,
    #                                                     '2': 2},
    #                                 decode_categorical_columns=True)

    # convert to tf.Dataset api
    ds = convert_x_y_to_tf_dataset(X, y, BATCH_SIZE)

    # extract input size
    input_size = x_train.shape[1]

    # initialize and train GAN model
    gan_model = GAN(input_size=input_size)
    c_loss_per_batch, c_loss_per_epoch, g_loss_per_batch, g_loss_per_epoch, c_acc_per_batch, c_acc_per_epoch = gan_model.train_gan(ds, BATCH_SIZE, N_EPOCHS)

    # generate plots
    plot_critic_generator_loss(list(range(1, len(c_loss_per_batch) + 1)), c_loss_per_batch, list(range(1, len(g_loss_per_batch) + 1)), g_loss_per_batch,
         "critic loss", "generator loss", "batch step #", "loss", "Critic and Generator loss values per batch step")
    plot_critic_generator_loss(list(range(1, len(c_loss_per_epoch) + 1)), c_loss_per_epoch, list(range(1, len(g_loss_per_epoch) + 1)), g_loss_per_epoch,
         "critic loss", "generator loss", "epoch #", "loss", "Critic and Generator loss values per epoch")

    plot_critic_accuracy(list(range(1, len(c_acc_per_epoch) + 1)), c_acc_per_epoch, "critic accuracy", "epoch #", "acc",
                         "Critic accuracy per epoch")

    # TODO
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

