from tensorflow.python.ops.numpy_ops import np_config

from SimpleClassifierForEvaluation import SimpleCLFForEvaluation
from WGAN import GANMonitor, WGAN
from WGAN2 import WGAN_test
from random_forest_model import *
from utils import plot_critic_generator_loss, plot_critic_accuracy

np_config.enable_numpy_behavior()  # TODO check


def train_wgan(ds, input_size, columns_size, num_classes, column_idx_to_scaler, column_idx_to_ohe, num_samples,
               columns, X_test, y_test):
    # Instantiate the WGAN model.
    wgan = WGAN(input_size, columns_size, num_classes)

    # configure checkpoint to save the critic and generator models during the training process
    checkpoint = tf.train.Checkpoint(generator_optimizer=wgan.generator_optimizer,
                                     critic_optimizer=wgan.critic_optimizer,
                                     generator=wgan.generator,
                                     critic=wgan.critic)

    # Instantiate GANMonitor Keras callback.
    evaluate_cbk = GANMonitor(column_idx_to_scaler=column_idx_to_scaler,
                              column_idx_to_ohe=column_idx_to_ohe,
                              checkpoint=checkpoint,
                              num_samples=num_samples,
                              columns=columns,
                              X_test=X_test,
                              y_test=y_test)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TF_LOGS_PATH)

    # Start training the model
    history = wgan.fit(ds, batch_size=BATCH_SIZE, epochs=N_EPOCHS, callbacks=[evaluate_cbk, tensorboard_callback])

    # generate plots
    critic_loss = history.history['generator_loss']
    generator_loss = history.history['critic_loss']
    plot_critic_generator_loss(list(range(1, len(critic_loss) + 1)), critic_loss, list(range(1, len(generator_loss) + 1)), generator_loss,
         "critic loss", "generator loss", "epoch #", "loss", "Critic and Generator loss values per epoch")


def classifier_evaluation(labels_to_num_dict: dict,
                          data_path: str):
    print(f"Train RandomForestClassifier model on {data_path} data and evaluate")
    model = SimpleCLFForEvaluation(labels_to_num_dict)
    model.train_and_score_model()
    print()

    print(f"Train LogisticRegression model on {data_path} data and evaluate")
    model = SimpleCLFForEvaluation(labels_to_num_dict, model_type='LogisticRegression')
    model.train_and_score_model()
    print()

    return model.X_test, model.y_test


def section1():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    num_classes = 2
    diabetes_columns_size = [1] * 8
    german_credit_columns_size = [1, 1, 1, 1, 1, 1, 1, 4, 5, 10, 5, 5, 4, 3, 4, 3, 3, 4, 2, 2]

    # diabetes dataset
    diabetes_labels_to_num_dict = {'tested_positive': 1, 'tested_negative': -1}
    X, y, column_idx_to_scaler, column_idx_to_ohe = read_and_prepare_dataset(path_to_arff_file=DIABETES_PATH,
                                                                             labels_to_num_dict=diabetes_labels_to_num_dict,
                                                                             decode_categorical_columns=True)

    # # german_credit dataset
    # german_credit_labels_to_num_dict = {'1': -1, '2': 1}
    # X, y, column_idx_to_scaler, column_idx_to_ohe = read_and_prepare_dataset(path_to_arff_file=GERMAN_CREDIT_PATH,
    #                                                                          labels_to_num_dict=german_credit_labels_to_num_dict,
    #                                                                          decode_categorical_columns=True)

    # convert to tf.Dataset api
    ds = convert_x_y_to_tf_dataset(X, y, batch_size=BATCH_SIZE, include_y=True)

    # extract input size
    input_size = X.shape[1]

    # classifier evaluation
    X_test, y_test = classifier_evaluation(diabetes_labels_to_num_dict, DIABETES_PATH)

    # train wgan model on german_credit dataset
    train_wgan(ds, input_size, diabetes_columns_size, num_classes, column_idx_to_scaler, column_idx_to_ohe,
               768, X.columns.tolist(), X_test, y_test)

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
    #print(wgan.generate_samples(3, 1, columns_size))

    # # german_credit dataset
    # X, y, ohe = read_and_prepare_dataset(path_to_arff_file=GERMAN_CREDIT_PATH,
    #                                      labels_to_num_dict={'1': 1,
    #                                                          '2': 2},
    #                                      decode_categorical_columns=True)



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

