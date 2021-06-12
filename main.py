from keras.models import load_model
from table_evaluator import TableEvaluator
from tensorflow.python.ops.numpy_ops import np_config

from CGAN import CGAN
from CWGAN import CWGAN
from SimpleClassifierForEvaluation import SimpleCLFForEvaluation
from random_forest_model import *
from utils import plot_loss_history, GanSampleGenerator, plot_accuracy_history, log, model_confidence_score_distribution

np_config.enable_numpy_behavior()


def part1_section3(experiment_dir, gan_sample_generator):
    # load models
    generator = load_model(f"{experiment_dir}/generator.h5")
    critic = load_model(f"{experiment_dir}/critic.h5")

    samples, generated_samples, labels_input = gan_sample_generator.generate_samples(generator, random_latent_noise=True)
    # extract 100 random samples
    generated_samples_reduced = np.array(generated_samples)[:100, :]
    labels_input_reduced = labels_input[:100]

    # create inverted labels for the fake samples
    y = np.ones((100, 1))

    if IS_LABEL_CONDITIONAL:
        accuracy = critic.evaluate([generated_samples_reduced, labels_input_reduced], y)[1]
        pred = critic.predict([generated_samples_reduced, labels_input_reduced])
    else:
        accuracy = critic.evaluate(generated_samples_reduced, y)[1]
        pred = critic.predict([generated_samples_reduced, labels_input_reduced])

    # extract samples that fooled the critic
    samples_that_fooled_the_critic = np.array(samples)[np.argwhere(pred > 0.7)[:, 0].tolist()]
    # extract samples that not fooled the critic
    samples_that_not_fooled_the_critic = np.array(samples)[np.argwhere(pred < 0.5)[:, 0].tolist()]

    return accuracy, samples_that_fooled_the_critic, samples_that_not_fooled_the_critic


def train_cgan(ds, df_real, input_size, columns_size, num_classes,  column_idx_to_scaler, column_idx_to_ohe, num_samples,
               df_columns, X_test, y_test, num_positive_negative_classes, positive_negative_labels, experiment_dir, logger):
    cgan = CGAN(input_size, columns_size, num_classes, is_label_conditional=IS_LABEL_CONDITIONAL)
    gan_sample_generator = GanSampleGenerator(LATENT_NOISE_SIZE,
                                              column_idx_to_scaler,
                                              column_idx_to_ohe,
                                              columns_size,
                                              num_samples,
                                              is_label_conditional=IS_LABEL_CONDITIONAL,
                                              num_positive_negative_classes=num_positive_negative_classes,
                                              evaluation_mode=True,
                                              positive_negative_labels=positive_negative_labels)
    d_loss1_epoch, d_loss2_epoch, g_loss_epoch, d_acc1_epoch, d_acc2_epoch, max_score_for_fixed_latent_noise, \
    max_score_for_random_latent_noise, samples, generated_samples, labels = cgan.train(ds,
                                                                                       BATCH_SIZE,
                                                                                       gan_sample_generator,
                                                                                       X_test, y_test,
                                                                                       N_EPOCHS,
                                                                                       df_columns,
                                                                                       experiment_dir,
                                                                                       logger)

    # table evaluation
    target_column_name = df_real.columns[-1]
    df_fake = pd.DataFrame(data=np.concatenate((np.array(samples), labels.reshape(-1, 1)), axis=1),
                           columns=df_columns + [target_column_name])
    table_evaluator = TableEvaluator(df_real, df_fake)
    #table_evaluator.visual_evaluation() TODO
    logger.info(table_evaluator.evaluate(target_col=target_column_name))

    # save fake dataframe
    df_fake.to_csv(f"{experiment_dir}/df_fake.csv", index=False)

    # line plots of loss
    plot_loss_history(d_loss1_epoch, d_loss2_epoch, g_loss_epoch, experiment_dir)

    # line plots of accuracy
    plot_accuracy_history(d_acc1_epoch, d_acc2_epoch, experiment_dir)

    logger.info("")
    logger.info("Best ML efficacy score fixed latent noise: {}, random latent noise: {}".format(
        max_score_for_fixed_latent_noise,
        max_score_for_random_latent_noise))

    # part 1 section 3
    accuracy, samples_that_fooled_the_critic, samples_that_not_fooled_the_critic = part1_section3(experiment_dir, gan_sample_generator)

    logger.info("")
    logger.info("100 random generated samples were able to achieve {} accuracy".format(accuracy))
    logger.info("Samples that fooled the critic: {}".format(samples_that_fooled_the_critic))
    logger.info("Samples that not fooled the critic: {}".format(samples_that_not_fooled_the_critic))


def train_cwgan(X, y, df_real, input_size, columns_size, num_classes, column_idx_to_scaler, column_idx_to_ohe, num_samples,
                df_columns, X_test, y_test, num_positive_negative_classes, positive_negative_labels, experiment_dir, logger):
    gan = CWGAN(input_size, columns_size, num_classes,
                is_label_conditional=IS_LABEL_CONDITIONAL, positive_negative_labels=positive_negative_labels)
    gan_sample_generator = GanSampleGenerator(LATENT_NOISE_SIZE,
                                              column_idx_to_scaler,
                                              column_idx_to_ohe,
                                              columns_size,
                                              num_samples,
                                              is_label_conditional=IS_LABEL_CONDITIONAL,
                                              num_positive_negative_classes=num_positive_negative_classes,
                                              evaluation_mode=True,
                                              positive_negative_labels=positive_negative_labels)
    c1_hist, c2_hist, g_hist, max_score_for_fixed_latent_noise, max_score_for_random_latent_noise, samples, generated_samples, labels = gan.train(X.to_numpy(), y.to_numpy(), BATCH_SIZE, gan_sample_generator,
        X_test, y_test, N_EPOCHS, df_columns, experiment_dir, logger)

    # table evaluation
    target_column_name = df_real.columns[-1]
    df_fake = pd.DataFrame(data=np.concatenate((np.array(samples), labels.reshape(-1, 1)), axis=1),
                           columns=df_columns + [target_column_name])
    table_evaluator = TableEvaluator(df_real, df_fake)
    #table_evaluator.visual_evaluation() TODO
    logger.info(table_evaluator.evaluate(target_col=target_column_name))

    # save fake dataframe
    df_fake.to_csv(f"{experiment_dir}/df_fake.csv", index=False)

    # line plots of loss
    plot_loss_history(c1_hist, c2_hist, g_hist, experiment_dir)

    logger.info("")
    logger.info("Best ML efficacy score fixed latent noise: {}, random latent noise: {}".format(
        max_score_for_fixed_latent_noise,
        max_score_for_random_latent_noise))

# def train_gan(ds, input_size, columns_size, num_classes, column_idx_to_scaler, column_idx_to_ohe, num_samples,
#                columns, X_test, y_test):
#     # Instantiate the WGAN model.
#     gan = GAN(input_size, columns_size, num_classes)
#
#     # # configure checkpoint to save the critic and generator models during the training process
#     # checkpoint = tf.train.Checkpoint(generator_optimizer=wgan.generator_optimizer,
#     #                                  critic_optimizer=wgan.critic_optimizer,
#     #                                  generator=wgan.generator,
#     #                                  critic=wgan.critic)
#     #
#     # # Instantiate GANMonitor Keras callback.
#     # evaluate_cbk = GANMonitor(column_idx_to_scaler=column_idx_to_scaler,
#     #                           column_idx_to_ohe=column_idx_to_ohe,
#     #                           checkpoint=checkpoint,
#     #                           num_samples=num_samples,
#     #                           columns=columns,
#     #                           X_test=X_test,
#     #                           y_test=y_test)
#     #
#     # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TF_LOGS_PATH)
#
#     # Start training the model
#     c_loss_per_batch, c_loss_per_epoch, g_loss_per_batch, g_loss_per_epoch, c_acc_per_batch, c_acc_per_epoch = gan.train_gan(ds, BATCH_SIZE, N_EPOCHS)
#
#     # generate plots
#     plot_critic_generator_loss(list(range(1, len(c_loss_per_batch) + 1)), c_loss_per_batch, list(range(1, len(g_loss_per_batch) + 1)), g_loss_per_batch,
#          "critic loss", "generator loss", "batch step #", "loss", "Critic and Generator loss values per batch step")
#     plot_critic_generator_loss(list(range(1, len(c_loss_per_epoch) + 1)), c_loss_per_epoch, list(range(1, len(g_loss_per_epoch) + 1)), g_loss_per_epoch,
#          "critic loss", "generator loss", "epoch #", "loss", "Critic and Generator loss values per epoch")
#
#     plot_critic_accuracy(list(range(1, len(c_acc_per_epoch) + 1)), c_acc_per_epoch, "critic accuracy", "epoch #", "acc",
#                          "Critic accuracy per epoch")
#
#     # generate plots
#     # critic_loss = history.history['generator_loss']
#     # generator_loss = history.history['critic_loss']
#     # plot_critic_generator_loss(list(range(1, len(critic_loss) + 1)), critic_loss, list(range(1, len(generator_loss) + 1)), generator_loss,
#     #      "critic loss", "generator loss", "epoch #", "loss", "Critic and Generator loss values per epoch")

#
# def train_wgan(ds, input_size, columns_size, num_classes, column_idx_to_scaler, column_idx_to_ohe, num_samples,
#                columns, X_test, y_test):
#     # Instantiate the WGAN model.
#     wgan = WGAN(input_size, columns_size, num_classes)
#
#     # configure checkpoint to save the critic and generator models during the training process
#     checkpoint = tf.train.Checkpoint(generator_optimizer=wgan.generator_optimizer,
#                                      critic_optimizer=wgan.critic_optimizer,
#                                      generator=wgan.generator,
#                                      critic=wgan.critic)
#
#     # Instantiate GANMonitor Keras callback.
#     evaluate_cbk = GANMonitor(column_idx_to_scaler=column_idx_to_scaler,
#                               column_idx_to_ohe=column_idx_to_ohe,
#                               checkpoint=checkpoint,
#                               num_samples=num_samples,
#                               columns=columns,
#                               X_test=X_test,
#                               y_test=y_test)
#
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TF_LOGS_PATH)
#
#     # Start training the model
#     history = wgan.fit(ds, batch_size=BATCH_SIZE, epochs=N_EPOCHS, callbacks=[evaluate_cbk, tensorboard_callback])
#
#     # generate plots
#     critic_loss = history.history['generator_loss']
#     generator_loss = history.history['critic_loss']
#     plot_critic_generator_loss(list(range(1, len(critic_loss) + 1)), critic_loss, list(range(1, len(generator_loss) + 1)), generator_loss,
#          "critic loss", "generator loss", "epoch #", "loss", "Critic and Generator loss values per epoch")
#

def classifier_evaluation(labels_to_num_dict: dict,
                          data_path: str,
                          logger):
    logger.info(f"Train RandomForestClassifier model on {data_path} data and evaluate")
    model = SimpleCLFForEvaluation(labels_to_num_dict)
    score = model.train_and_score_model()
    logger.info(f'model score on real data: {score}')
    logger.info("")

    model_confidence_score_distribution(model)

    logger.info(f"Train LogisticRegression model on {data_path} data and evaluate")
    model = SimpleCLFForEvaluation(labels_to_num_dict, model_type='LogisticRegression')
    score = model.train_and_score_model()
    logger.info(f'model score on real data: {score}')
    logger.info("")

    return model.X_test, model.y_test


def section1():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    num_classes = 2

    # diabetes dataset
    diabetes_labels_to_num_dict = {'tested_positive': 1 if GAN_MODE == 'cgan' else -1, 'tested_negative': 0 if GAN_MODE == 'cgan' else 1}
    diabetes_columns_size = [1] * 8
    diabetes_num_positive_negative_classes = (500, 268)

    # german_credit dataset
    german_credit_labels_to_num_dict = {'1': 0 if GAN_MODE == 'cgan' else 1, '2': 1 if GAN_MODE == 'cgan' else -1}
    german_credit_columns_size = [1, 1, 1, 1, 1, 1, 1, 4, 5, 10, 5, 5, 4, 3, 4, 3, 3, 4, 2, 2]
    german_credit_num_positive_negative_classes = (700, 300)

    if DATASET == 'diabetes':
        dataset_path = DIABETES_PATH
        labels_to_num_dict = diabetes_labels_to_num_dict
        columns_size = diabetes_columns_size
        num_positive_negative_classes = diabetes_num_positive_negative_classes
    else:
        dataset_path = GERMAN_CREDIT_PATH
        labels_to_num_dict = german_credit_labels_to_num_dict
        columns_size = german_credit_columns_size
        num_positive_negative_classes = german_credit_num_positive_negative_classes

    X, y, column_idx_to_scaler, column_idx_to_ohe = read_and_prepare_dataset(path_to_arff_file=dataset_path,
                                                                             labels_to_num_dict=labels_to_num_dict,
                                                                             decode_categorical_columns=True)

    # convert to tf.Dataset api
    ds = convert_x_y_to_tf_dataset(X, y, batch_size=BATCH_SIZE, include_y=True)

    # extract input size
    input_size = X.shape[1]

    # extract number of training samples
    num_samples = X.shape[0]

    experiment_name = "mode={}_epochs={}_batch={}_c_lr={}_g_lr={}_is_conditional={}_c_steps={}_c_dropout={}_noise_size={}_seed={}".format(
        GAN_MODE,
        N_EPOCHS,
        BATCH_SIZE,
        CRITIC_LR,
        GENERATOR_LR,
        IS_LABEL_CONDITIONAL,
        CRITIC_STEPS,
        CRITIC_DROPOUT,
        LATENT_NOISE_SIZE,
        SEED)

    experiment_dir = os.sep.join([DATASET, experiment_name])
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    log_filename = os.sep.join([experiment_dir, f'{experiment_name}.txt'])

    logger = log(".", log_filename)
    logger.info("#################################################################")

    # classifier evaluation
    X_test, y_test = classifier_evaluation(labels_to_num_dict, dataset_path, logger)

    if GAN_MODE == 'cgan':
        positive_negative_labels = [0, 1]
        train_cgan(ds, pd.concat([X, y], axis=1), input_size, columns_size, num_classes, column_idx_to_scaler, column_idx_to_ohe,
                   num_samples, X.columns.tolist(), X_test, y_test, num_positive_negative_classes, positive_negative_labels, experiment_dir, logger)
    else:
        positive_negative_labels = [1, -1]
        train_cwgan(X, y, input_size, columns_size, num_classes, column_idx_to_scaler, column_idx_to_ohe,
                   num_samples, X.columns.tolist(), X_test, y_test, num_positive_negative_classes, positive_negative_labels, experiment_dir, logger)


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

