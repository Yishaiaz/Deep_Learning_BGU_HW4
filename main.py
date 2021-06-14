import matplotlib.pyplot as plt
from keras.models import load_model
from table_evaluator import TableEvaluator
from tensorflow.python.ops.numpy_ops import np_config

from CGAN import CGAN
from CWGAN import CWGAN
from SimpleClassifierForEvaluation import SimpleCLFForEvaluation
from gan_with_twist import GANBBModel
from random_forest_model import *
from utils import plot_loss_history, GanSampleGenerator, plot_accuracy_history, log, \
    model_confidence_score_distribution, generate_and_draw_boxplots, \
    real_to_generated_distance, invert_labels_to_num_dict

np_config.enable_numpy_behavior()


def part1_section3(num_of_random_samples, experiment_dir, gan_sample_generator, df_fake: pd.DataFrame, df_real_not_normalized: pd.DataFrame):
    df_real_not_normalized = df_real_not_normalized.iloc[:, :-1]
    # Gather numeric and categorical columns into a list
    numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(df_real_not_normalized)
    numeric_columns = numeric_columns.tolist()
    categorical_columns = categorical_columns.tolist()

    # load models
    generator = load_model(f"{experiment_dir}/generator.h5")
    critic = load_model(f"{experiment_dir}/critic.h5")

    samples, generated_samples, labels_input = gan_sample_generator.generate_samples(generator, random_latent_noise=True)

    # extract random samples
    generated_samples_reduced = np.array(generated_samples)[:num_of_random_samples, :]
    labels_input_reduced = labels_input[:num_of_random_samples]
    samples_reduced = np.array(samples)[:num_of_random_samples, :]

    df_samples_reduced = pd.DataFrame(data=np.concatenate((np.array(samples_reduced), labels_input_reduced.reshape(-1, 1)), axis=1),
                                      columns=df_fake.columns.tolist())
    df_samples_reduced[numeric_columns] = df_samples_reduced[numeric_columns].apply(pd.to_numeric)

    # create inverted labels for the fake samples
    y = np.ones((num_of_random_samples, 1))

    if IS_LABEL_CONDITIONAL:
        accuracy = critic.evaluate([generated_samples_reduced, labels_input_reduced], y)[1]
        pred = critic.predict([generated_samples_reduced, labels_input_reduced])
    else:
        accuracy = critic.evaluate(generated_samples_reduced, y)[1]
        pred = critic.predict([generated_samples_reduced, labels_input_reduced])

    # extract samples that fooled the critic
    samples_that_fooled_the_critic = np.array(samples)[np.argwhere(pred > 0.5)[:, 0].tolist()]
    # extract samples that not fooled the critic
    samples_that_not_fooled_the_critic = np.array(samples)[np.argwhere(pred < 0.5)[:, 0].tolist()]

    # boxplots
    generate_and_draw_boxplots(experiment_dir, df_samples_reduced[numeric_columns], df_real=df_real_not_normalized[numeric_columns], num_of_samples=num_of_random_samples)

    # distances between real and fake
    column_correlation, euclidean_distance = real_to_generated_distance(real_df=df_real_not_normalized, fake_df=df_fake.iloc[:, :-1], categorical_columns=categorical_columns)

    return accuracy, samples_that_fooled_the_critic, samples_that_not_fooled_the_critic, column_correlation, euclidean_distance


def train_cgan(ds, df_real, input_size, columns_size, num_classes, column_idx_to_scaler, column_idx_to_ohe, num_samples,
               X_test, y_test, num_positive_negative_classes, positive_negative_labels, experiment_dir, logger, df_real_not_normalized):
    df_columns = df_real_not_normalized.iloc[:, :-1].columns.tolist()
    # Gather numeric and categorical columns into a list
    numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(df_real_not_normalized)

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
                                                                                       df_real_not_normalized.iloc[:, :-1],
                                                                                       experiment_dir,
                                                                                       logger)

    # table evaluation
    target_column_name = df_real.columns[-1]
    df_fake = pd.DataFrame(data=np.concatenate((np.array(samples), labels.reshape(-1, 1)), axis=1),
                           columns=df_columns + [target_column_name])
    df_fake[numeric_columns] = df_fake[numeric_columns].apply(pd.to_numeric)
    table_evaluator = TableEvaluator(df_real_not_normalized, df_fake, cat_cols=categorical_columns.tolist())
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
    logger.info("")

    # part 1 section 3
    accuracy, samples_that_fooled_the_critic, samples_that_not_fooled_the_critic, \
    column_correlation, euclidean_distance = part1_section3(NUM_OF_RANDOM_SAMPLES, experiment_dir, gan_sample_generator, df_fake, df_real_not_normalized)

    logger.info("")
    logger.info("100 random generated samples were able to achieve {} accuracy".format(accuracy))
    logger.info("Samples that fooled the critic: {}".format(samples_that_fooled_the_critic))
    logger.info("Samples that not fooled the critic: {}".format(samples_that_not_fooled_the_critic))
    logger.info("Column correlation between fake and real data: {}".format(column_correlation))
    logger.info("Euclidean distance between fake and real data: {}".format(euclidean_distance))


def train_cwgan(X, y, df_real, input_size, columns_size, num_classes, column_idx_to_scaler, column_idx_to_ohe, num_samples,
                X_test, y_test, num_positive_negative_classes, positive_negative_labels, experiment_dir, logger, df_real_not_normalized):
    df_columns = df_real_not_normalized.iloc[:, :-1].columns.tolist()
    # Gather numeric and categorical columns into a list
    numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(df_real_not_normalized)

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
    c_loss1_hist, c_loss2_hist, g_loss_hist, max_score_for_fixed_latent_noise, max_score_for_random_latent_noise, samples, generated_samples, labels = gan.train(X.to_numpy(), y.to_numpy(), BATCH_SIZE, gan_sample_generator,
        X_test, y_test, N_EPOCHS, df_real_not_normalized.iloc[:, :-1], experiment_dir, logger)

    # table evaluation
    target_column_name = df_real.columns[-1]
    df_fake = pd.DataFrame(data=np.concatenate((np.array(samples), labels.reshape(-1, 1)), axis=1),
                           columns=df_columns + [target_column_name])
    df_fake[numeric_columns] = df_fake[numeric_columns].apply(pd.to_numeric)

    table_evaluator = TableEvaluator(df_real_not_normalized, df_fake, cat_cols=categorical_columns.tolist())
    #table_evaluator.visual_evaluation() TODO
    logger.info(table_evaluator.evaluate(target_col=target_column_name))

    # save fake dataframe
    df_fake.to_csv(f"{experiment_dir}/df_fake.csv", index=False)

    # line plots of loss
    plot_loss_history(c_loss1_hist, c_loss2_hist, g_loss_hist, experiment_dir)

    logger.info("")
    logger.info("Best ML efficacy score fixed latent noise: {}, random latent noise: {}".format(
        max_score_for_fixed_latent_noise,
        max_score_for_random_latent_noise))
    logger.info("")


def part_2_section_4_c(X_generated: np.array, confidence_scores: np.array,
                       clf: SimpleCLFForEvaluation,
                       labels_to_num_dict: dict,
                       experiment_dir: str = '',
                       number_of_bins: int = 5):
    probas_dist = clf.model_confidence_score_distribution(X_generated)
    dist_by_class = {}
    order_of_classes, n_classes = {position: class_value  for position, class_value in enumerate(clf.model.classes_)}, \
                                  len(clf.model.classes_)

    bins = np.linspace(0, 1, number_of_bins + 1)
    bins_absolute_errors = [[] for i in range(number_of_bins)]
    wrong_class_absolute_errors = [[] for i in range(number_of_bins)]
    inversed_labels_to_num_dict = invert_labels_to_num_dict(labels_to_num_dict)

    for pos, class_val in order_of_classes.items():
        dist_by_class[class_val] = []

    for row_idx, confidences_of_model in enumerate(probas_dist):
        class_predicted = order_of_classes[np.argmax(confidences_of_model)]
        confidence_of_predicted_class = confidences_of_model[np.argmax(confidences_of_model)]
        confidence_of_wrong_predicted_class = confidences_of_model[np.argmin(confidences_of_model)]
        bin_idx_of_pred_confidence = np.digitize(confidence_of_predicted_class,
                                                 bins=bins) - 1 if confidence_of_predicted_class < 1 else number_of_bins - 1
        bin_idx_of_wrong_pred_confidence = np.digitize(confidence_of_wrong_predicted_class,
                                                       bins=bins) - 1 if confidence_of_wrong_predicted_class < 1 else number_of_bins - 1

        confidence_given_to_gan = confidence_scores[row_idx]

        bins_absolute_errors[bin_idx_of_pred_confidence] += [np.abs(confidence_of_predicted_class - confidence_given_to_gan)]
        wrong_class_absolute_errors[bin_idx_of_wrong_pred_confidence] += [np.abs(confidence_of_wrong_predicted_class - confidence_given_to_gan)]

        dist_by_class[class_predicted] += [[confidence_of_predicted_class, confidence_given_to_gan]]

    fig, axis = plt.subplots((n_classes))
    # plot confidences of each class
    for idx, class_value in enumerate(order_of_classes.values()):
        confidence_vals = np.asarray(dist_by_class[class_value])
        abs_error_between_confidence = np.abs(confidence_vals[:, 0] - confidence_vals[:, 1])

        axis[idx].plot(np.arange(0, len(abs_error_between_confidence), 1), abs_error_between_confidence)

        axis[idx].set(ylim=(0, 1))
        class_label = inversed_labels_to_num_dict[class_value]
        axis[idx].set_title(f'Predicted Class={class_label}')

    fig.suptitle('Absolute Error between confidence\nof GAN and BB prediction confidence')
    plt.tight_layout()
    plt.legend()
    fig_path = os.sep.join([experiment_dir, 'absolute_error_of_confidence.png'])
    plt.savefig(fig_path)
    bins_mse = np.zeros(shape=(len(bins_absolute_errors), ))
    bins_wrong_mse = np.zeros(shape=(len(wrong_class_absolute_errors), ))
    for bin_idx, bin_absolute_errors in enumerate(bins_absolute_errors):
        bins_mse[bin_idx] = np.mean(bin_absolute_errors) if len(bin_absolute_errors) > 0 else 0
        wrong_bin_absolute_error = wrong_class_absolute_errors[bin_idx]
        bins_wrong_mse[bin_idx] = np.mean(wrong_bin_absolute_error) if len(wrong_bin_absolute_error) > 0 else 0

    fig, ax = plt.subplots()
    ax.bar(range(len(bins_mse)), bins_mse, label='MSE: Confidence of prediction', fc=(1, 0, 1, 0.6))
    ax.bar(range(len(bins_wrong_mse)), bins_wrong_mse, label='MSE: Confidence of wrong prediction', fc=(0, 1, 0, 0.4))
    ax.set_xticks(range(len(bins_mse)))
    ax.set_xticklabels([f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins)-1)])
    ax.set_ylim((0, 1))
    fig.suptitle('Mean Absolute Error between confidence\nof GAN and BB prediction confidence')
    plt.tight_layout()
    plt.legend()
    fig_path = os.sep.join([experiment_dir, 'mean_absolute_error_of_confidences.png'])
    plt.savefig(fig_path)

    fig, ax = plt.subplots()
    ax.hist(probas_dist[:, 0], edgecolor='black', label=f'{inversed_labels_to_num_dict[order_of_classes[0]]}',
            fc=(1, 1, 0, 0.5))
    ax.hist(probas_dist[:, 1], edgecolor='black', label=f'{inversed_labels_to_num_dict[order_of_classes[1]]}',
            fc=(1, 0, 1, 0.5))
    ax.set_title('All Confidence Scores of BB Model')
    plt.tight_layout()
    plt.legend()
    fig_path = os.sep.join([experiment_dir, 'confidence_scores_about_classes.png'])
    plt.savefig(fig_path)
    # plt.show()

    # calc which confidence intervals had lower MAE


def train_gan_with_twist(model, df_real, input_size, columns_size, column_idx_to_scaler, column_idx_to_ohe, num_samples,
                         positive_negative_labels, experiment_dir, logger, df_real_not_normalized):

    df_columns = df_real_not_normalized.iloc[:, :-1].columns.tolist()
    # Gather numeric and categorical columns into a list
    numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(df_real_not_normalized)

    gan_bb_model = GANBBModel(model, input_size, columns_size)
    gan_sample_generator = GanSampleGenerator(LATENT_NOISE_SIZE,
                                              column_idx_to_scaler,
                                              column_idx_to_ohe,
                                              columns_size,
                                              num_samples,
                                              is_label_conditional=False,
                                              evaluation_mode=True,
                                              positive_negative_labels=positive_negative_labels)
    d_loss1_epoch, d_loss2_epoch, g_loss_epoch, min_mse_score_for_fixed_latent_noise, \
    min_score_for_random_latent_noise, samples, generated_samples, labels = gan_bb_model.train(num_samples,
                                                                                               BATCH_SIZE,
                                                                                               gan_sample_generator,
                                                                                               N_EPOCHS,
                                                                                               experiment_dir,
                                                                                               logger)

    # table evaluation
    target_column_name = df_real.columns[-1]
    df_fake = pd.DataFrame(data=np.concatenate((np.array(samples), labels.reshape(-1, 1)), axis=1),
                           columns=df_columns + [target_column_name])
    df_fake[numeric_columns] = df_fake[numeric_columns].apply(pd.to_numeric)

    # save fake dataframe
    df_fake.to_csv(f"{experiment_dir}/df_fake.csv", index=False)

    # line plots of loss
    plot_loss_history(d_loss1_epoch, d_loss2_epoch, g_loss_epoch, experiment_dir)

    logger.info("")
    logger.info(
        "Best ML MSE score fixed latent noise: {}, random latent noise: {}".format(min_mse_score_for_fixed_latent_noise,
                                                                                   min_score_for_random_latent_noise))
    logger.info("")

    # part 2 section 4
    # accuracy, samples_that_fooled_the_critic, samples_that_not_fooled_the_critic, \
    # column_correlation, euclidean_distance = part1_section3(NUM_OF_RANDOM_SAMPLES, experiment_dir, gan_sample_generator, df_real)
    #
    # logger.info("")
    # logger.info("100 random generated samples were able to achieve {} accuracy".format(accuracy))
    # logger.info("Samples that fooled the critic: {}".format(samples_that_fooled_the_critic))
    # logger.info("Samples that not fooled the critic: {}".format(samples_that_not_fooled_the_critic))
    # logger.info("Column correlation between fake and real data: {}".format(column_correlation))
    # logger.info("Euclidean distance between fake and real data: {}".format(euclidean_distance))


def classifier_evaluation(labels_to_num_dict: dict,
                          data_path: str,
                          target_column_name,
                          logger):
    logger.info(f"Train RandomForestClassifier model on {data_path} data and evaluate")
    model = SimpleCLFForEvaluation(labels_to_num_dict, data_path=data_path)
    score = model.train_and_score_model()
    logger.info(f'model score on real data: {score}')
    logger.info("")

    # model_confidence_score_distribution(model, hue=target_column_name) TODO

    logger.info(f"Train LogisticRegression model on {data_path} data and evaluate")
    model = SimpleCLFForEvaluation(labels_to_num_dict, data_path=data_path, model_type='LogisticRegression')
    score = model.train_and_score_model()
    logger.info(f'model score on real data: {score}')
    logger.info("")

    return model.X_test, model.y_test


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    num_classes = 2

    # diabetes dataset
    diabetes_labels_to_num_dict = {'tested_positive': 1 if GAN_MODE == 'cgan' or GAN_MODE == 'gan_with_twist' else -1, 'tested_negative': 0 if GAN_MODE == 'cgan' or GAN_MODE == 'gan_with_twist' else 1}
    diabetes_columns_size = [1] * 8
    diabetes_num_positive_negative_classes = (500, 268)

    # german_credit dataset
    german_credit_labels_to_num_dict = {'1': 0 if GAN_MODE == 'cgan' or GAN_MODE == 'gan_with_twist' else 1, '2': 1 if GAN_MODE == 'cgan' else -1}
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

    # load given file into DataFrame
    df_real_not_normalized = read_arff_file_as_dataframe(dataset_path)
    # Gather numeric and categorical columns into a list
    numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(df_real_not_normalized)
    # iterate over all categorical columns and convert decode to string
    df_real_not_normalized = df_real_not_normalized.apply(lambda col: col.str.decode(encoding='UTF-8') if col.name in categorical_columns else col)
    df_real_not_normalized, df_real_not_normalized_y = df_real_not_normalized.iloc[:, :-1],\
                                                       df_real_not_normalized.iloc[:, -1:]

    # reorder columns
    numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(df_real_not_normalized)
    df_real_not_normalized = df_real_not_normalized[numeric_columns.tolist() + categorical_columns.tolist()]

    X, y, column_idx_to_scaler, column_idx_to_ohe, target_column_name = read_and_prepare_dataset(
        path_to_arff_file=dataset_path,
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

    experiment_dir = os.sep.join([DATASET, SECTION, experiment_name])
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    log_filename = os.sep.join([experiment_dir, f'{experiment_name}.txt'])

    logger = log(".", log_filename)
    logger.info("#################################################################")

    # classifier evaluation
    if GAN_MODE == 'gan_with_twist':
        random_forest_model = SimpleCLFForEvaluation(labels_to_num_dict, data_path=dataset_path)
        random_forest_model.train_model()
        X_test, y_test = random_forest_model.X_test, random_forest_model.y_test

        X_generated, confidences_given = X_test, np.random.random(len(X_test)) # TODO need to replace to generated samples and the confidence that was fed to the GAN
        part_2_section_4_c(X_generated, confidences_given, clf=random_forest_model, labels_to_num_dict=labels_to_num_dict, experiment_dir=experiment_dir)
    else:
        X_test, y_test = classifier_evaluation(labels_to_num_dict, dataset_path, target_column_name, logger)

    df_real_normalized = pd.concat([X, y], axis=1)
    df_real_not_normalized = pd.concat([df_real_not_normalized, y], axis=1)
    positive_negative_labels = [0, 1]

    if GAN_MODE == 'cgan':
        train_cgan(ds, df_real_normalized, input_size, columns_size, num_classes, column_idx_to_scaler, column_idx_to_ohe,
                   num_samples, X_test, y_test, num_positive_negative_classes, positive_negative_labels, experiment_dir, logger, df_real_not_normalized)
    elif GAN_MODE == 'cwgan':
        positive_negative_labels = [1, -1]
        train_cwgan(X, y, df_real_normalized, input_size, columns_size, num_classes, column_idx_to_scaler, column_idx_to_ohe,
                    num_samples, X_test, y_test, num_positive_negative_classes, positive_negative_labels, experiment_dir, logger, df_real_not_normalized)
    else:
        train_gan_with_twist(random_forest_model, df_real_normalized, input_size, columns_size, column_idx_to_scaler,
                             column_idx_to_ohe, num_samples, X.columns.tolist(), positive_negative_labels, experiment_dir, logger, df_real_not_normalized)


if __name__ == '__main__':
    main()

    global BATCH_SIZE
    global N_EPOCHS
    global GENERATOR_LR
    global CRITIC_LR
    global CRITIC_DROPOUT

    batch_sizes = [64, 128]
    n_epochs = [100, 500]
    generator_lr = [0.0005, 0.00005, 0.000005]
    critic_lr = [0.0005, 0.00005, 0.000005]
    critic_dropout = [0.2, 0.5]
    for batch_size in batch_sizes:
        for epoch in n_epochs:
            for dropout in critic_dropout:
                for g_lr in generator_lr:
                    for c_lr in critic_lr:
                        BATCH_SIZE = batch_size
                        N_EPOCHS = epoch
                        GENERATOR_LR = g_lr
                        CRITIC_LR = c_lr
                        CRITIC_DROPOUT = dropout

                        main()

    # input_size = list(ds.take(1).as_numpy_iterator())[0].shape[0]
    # generator = Generator(input_size=input_size)
    # discriminator = Discriminator(input_size=input_size)
    # generated = generator.generate_sample()
    # random_real_sample = tf.convert_to_tensor(list(ds.take(1).as_numpy_iterator())[0])
    # fake_decision, real_decision = discriminator.test_generator_output(generator_sample=generated.reshape(1, -1),
    #                                                                    real_sample=random_real_sample.reshape(1, -1))

