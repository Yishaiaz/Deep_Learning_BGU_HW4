from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.ops.numpy_ops import np_config

from WGAN import WGAN
from random_forest_model import *
from utils import plot_critic_generator_loss, plot_critic_accuracy

np_config.enable_numpy_behavior()  # TODO check


def section1():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # diabetes dataset
    X, y, _ = read_and_prepare_dataset(path_to_arff_file=DIABETES_PATH,
                                       labels_to_num_dict={'tested_positive': 1,
                                                           'tested_negative': -1},
                                       decode_categorical_columns=True)

    # convert to tf.Dataset api
    ds = convert_x_y_to_tf_dataset(X, y, batch_size=BATCH_SIZE, include_y=True)

    # extract input size
    input_size = X.shape[1]

    # Instantiate the customer `GANMonitor` Keras callback.
    # cbk = GANMonitor(num_img=3, latent_dim=noise_dim)

    # Instantiate the WGAN model.
    wgan = WGAN(input_size, [1] * input_size, 2)

    # Start training the model
    wgan.fit(ds, batch_size=BATCH_SIZE, epochs=N_EPOCHS)

    # initialize and train GAN model for diabetes dataset
    # gan_model = WGAN_GP(input_size=input_size, columns_size=[1] * input_size, num_classes=2)
    # c_loss_per_batch, c_loss_per_epoch, g_loss_per_batch, g_loss_per_epoch, c_acc_per_batch, c_acc_per_epoch = gan_model.train_gan(ds, BATCH_SIZE, N_EPOCHS)
    #
    # # generate plots
    # plot_critic_generator_loss(list(range(1, len(c_loss_per_batch) + 1)), c_loss_per_batch, list(range(1, len(g_loss_per_batch) + 1)), g_loss_per_batch,
    #      "critic loss", "generator loss", "batch step #", "loss", "Critic and Generator loss values per batch step")
    # plot_critic_generator_loss(list(range(1, len(c_loss_per_epoch) + 1)), c_loss_per_epoch, list(range(1, len(g_loss_per_epoch) + 1)), g_loss_per_epoch,
    #      "critic loss", "generator loss", "epoch #", "loss", "Critic and Generator loss values per epoch")
    #
    # plot_critic_accuracy(list(range(1, len(c_acc_per_epoch) + 1)), c_acc_per_epoch, "critic accuracy", "epoch #", "acc",
    #                      "Critic accuracy per epoch")

    # TODO
    print(wgan.generate_samples(3, 0))

    # german_credit dataset
    X, y, ohe = read_and_prepare_dataset(path_to_arff_file=GERMAN_CREDIT_PATH,
                                         labels_to_num_dict={'1': 1,
                                                             '2': 2},
                                         decode_categorical_columns=True)



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

