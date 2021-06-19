import matplotlib.pyplot as plt
from keras.models import load_model
from pandas_profiling import ProfileReport
from table_evaluator import TableEvaluator
from tensorflow.python.ops.numpy_ops import np_config

from CGAN import CGAN
from CWGAN import CWGAN
from SimpleClassifierForEvaluation import SimpleCLFForEvaluation
from gan_with_twist import GANBBModel
from gan_with_twist_binary_cross_entropy import GANBBModelBinaryCE
from random_forest_model import *
from utils import plot_loss_history, GanSampleGenerator, plot_accuracy_history, log, \
    model_confidence_score_distribution, generate_and_draw_boxplots, \
    real_to_generated_distance, invert_labels_to_num_dict, GanWithTwistSampleGenerator, plot_critic_generator_loss, \
    plot_score_metrics

np_config.enable_numpy_behavior()


def part1_section3(generator, critic, num_of_random_samples, experiment_dir, gan_sample_generator,
                   df_fake: pd.DataFrame, df_real_not_normalized: pd.DataFrame):
    df_real_not_normalized = df_real_not_normalized.iloc[:, :-1]
    # Gather numeric and categorical columns into a list
    numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(df_real_not_normalized)
    numeric_columns = numeric_columns.tolist()
    categorical_columns = categorical_columns.tolist()

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
    generate_and_draw_boxplots(experiment_dir, df_samples_reduced[numeric_columns],
                               df_real=df_real_not_normalized[numeric_columns].iloc[:num_of_random_samples, :])

    # distances between real and fake
    column_correlation, euclidean_distance = real_to_generated_distance(real_df=df_real_not_normalized,
                                                                        fake_df=df_fake.iloc[:, :-1],
                                                                        categorical_columns=categorical_columns,
                                                                        numeric_columns=numeric_columns)

    return accuracy, samples_that_fooled_the_critic, samples_that_not_fooled_the_critic, column_correlation, euclidean_distance


def train_cgan_and_generate_statistics(ds, df_real, input_size, columns_size, num_classes, column_idx_to_scaler, column_idx_to_ohe, num_samples,
                                       X_test, y_test, num_positive_negative_classes, positive_negative_labels, experiment_dir, logger, df_real_not_normalized):
    df_columns = df_real_not_normalized.iloc[:, :-1].columns.tolist()
    # Gather numeric and categorical columns into a list
    numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(df_real_not_normalized)

    cgan = CGAN(input_size, columns_size, num_classes, is_label_conditional=IS_LABEL_CONDITIONAL, generator_lr=GENERATOR_LR, discriminator_lr=CRITIC_LR, discriminator_dropout=CRITIC_DROPOUT)
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
    max_score_for_random_latent_noise, samples, generated_samples, labels, score_for_fixed_latent_noise_hist, score_for_random_latent_noise_hist = cgan.train(
        ds,
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

    # line plot of scores
    plot_score_metrics(list(range(1, len(score_for_fixed_latent_noise_hist) + 1)), score_for_fixed_latent_noise_hist,
                       list(range(1, len(score_for_random_latent_noise_hist) + 1)), score_for_random_latent_noise_hist, "fixed latent noise", "random latent noise", "epoch", "score", "ML_efficacy_scores", experiment_dir)

    logger.info("")
    logger.info("Best ML efficacy score fixed latent noise: {}, random latent noise: {}".format(
        max_score_for_fixed_latent_noise,
        max_score_for_random_latent_noise))
    logger.info("")

    # load models
    generator = load_model(f"{experiment_dir}/generator.h5")
    critic = load_model(f"{experiment_dir}/critic.h5")

    # part 1 section 3
    accuracy, samples_that_fooled_the_critic, samples_that_not_fooled_the_critic, \
    column_correlation, euclidean_distance = part1_section3(generator, critic, NUM_OF_RANDOM_SAMPLES_PART1, experiment_dir, gan_sample_generator, df_fake, df_real_not_normalized)

    logger.info("")
    logger.info("100 random generated samples were able to achieve {} accuracy".format(accuracy))
    logger.info("Samples that fooled the critic: {}".format(samples_that_fooled_the_critic))
    logger.info("Samples that not fooled the critic: {}".format(samples_that_not_fooled_the_critic))
    logger.info("Column correlation between fake and real data: {}".format(column_correlation))
    logger.info("Euclidean distance between fake and real data: {}".format(euclidean_distance))


def train_cwgan_and_generate_statistics(X, y, df_real, input_size, columns_size, num_classes, column_idx_to_scaler, column_idx_to_ohe, num_samples,
                                        X_test, y_test, num_positive_negative_classes, positive_negative_labels, experiment_dir, logger, df_real_not_normalized):
    df_columns = df_real_not_normalized.iloc[:, :-1].columns.tolist()
    # Gather numeric and categorical columns into a list
    numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(df_real_not_normalized)

    gan = CWGAN(input_size, columns_size, num_classes,
                is_label_conditional=IS_LABEL_CONDITIONAL, positive_negative_labels=positive_negative_labels,
                generator_lr=GENERATOR_LR, critic_lr=CRITIC_LR, critic_dropout=CRITIC_DROPOUT)
    gan_sample_generator = GanSampleGenerator(LATENT_NOISE_SIZE,
                                              column_idx_to_scaler,
                                              column_idx_to_ohe,
                                              columns_size,
                                              num_samples,
                                              is_label_conditional=IS_LABEL_CONDITIONAL,
                                              num_positive_negative_classes=num_positive_negative_classes,
                                              evaluation_mode=True,
                                              positive_negative_labels=positive_negative_labels)
    c_loss1_hist, c_loss2_hist, g_loss_hist, max_score_for_fixed_latent_noise, max_score_for_random_latent_noise,\
    samples, generated_samples, labels, score_for_fixed_latent_noise_hist, score_for_random_latent_noise_hist = gan.train(X.to_numpy(), y.to_numpy(), BATCH_SIZE, gan_sample_generator,
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

    # line plot of scores
    plot_score_metrics(list(range(1, len(score_for_fixed_latent_noise_hist) + 1)), score_for_fixed_latent_noise_hist,
                       list(range(1, len(score_for_random_latent_noise_hist) + 1)), score_for_random_latent_noise_hist, "fixed latent noise", "random latent noise", "epoch", "score", "ML_efficacy_scores", experiment_dir)

    logger.info("")
    logger.info("Best ML efficacy score fixed latent noise: {}, random latent noise: {}".format(
        max_score_for_fixed_latent_noise,
        max_score_for_random_latent_noise))
    logger.info("")


def part_2_section_4_c(X_generated: np.array, confidence_scores: np.array,
                       clf: SimpleCLFForEvaluation,
                       labels_to_num_dict: dict,
                       experiment_dir: str = '',
                       number_of_bins: int = 5,
                       proba_idx_to_measure: int = 1):
    probas_dist = clf.model_confidence_score_distribution(X_generated)
    dist_by_class = {}
    order_of_classes, n_classes = {position: class_value for position, class_value in enumerate(clf.model.classes_)}, \
                                  len(clf.model.classes_)

    bins = np.linspace(0, 1, number_of_bins + 1)
    bins_absolute_errors = [[] for i in range(number_of_bins)]
    absolute_errors_between_confidences = []
    inverse_labels_to_num_dict = invert_labels_to_num_dict(labels_to_num_dict)

    for pos, class_val in order_of_classes.items():
        dist_by_class[class_val] = []

    for row_idx, confidences_of_model in enumerate(probas_dist):
        confidence_of_predicted_class = confidences_of_model[proba_idx_to_measure]
        bin_idx_of_pred_confidence = np.digitize(confidence_of_predicted_class,
                                                 bins=bins) - 1 if confidence_of_predicted_class < 1 else number_of_bins - 1

        confidence_given_to_gan = confidence_scores[row_idx]
        absolute_confidence_error = np.abs(confidence_of_predicted_class - confidence_given_to_gan)
        absolute_errors_between_confidences += [absolute_confidence_error]
        bins_absolute_errors[bin_idx_of_pred_confidence] += [absolute_confidence_error]


    # plot confidences absolute error
    fig, ax = plt.subplots()
    ax.scatter(np.arange(0, len(absolute_errors_between_confidences), 1), absolute_errors_between_confidences)

    ax.set_ylim((0, 1))
    fig.suptitle(f'Absolute Error between confidence of class {order_of_classes[proba_idx_to_measure]}\nof BB prediction GAN and confidence')
    plt.tight_layout()
    fig_path = os.sep.join([experiment_dir, f'absolute_error_of_confidence_class{order_of_classes[proba_idx_to_measure]}.png'])
    plt.savefig(fig_path)

    bins_mse = np.zeros(shape=(len(bins_absolute_errors), ))

    for bin_idx, bin_absolute_errors in enumerate(bins_absolute_errors):
        bins_mse[bin_idx] = np.mean(bin_absolute_errors) if len(bin_absolute_errors) > 0 else 0

    fig, ax = plt.subplots()
    ax.bar(range(len(bins_mse)), bins_mse, label='MSE: Confidence of prediction', fc=(1, 0, 1, 0.6))

    ax.set_xlabel(f'confidence for class {order_of_classes[proba_idx_to_measure]}')
    ax.set_ylabel('MSE for confidence given to GAN & scored by BB')
    ax.set_xticks(range(len(bins_mse)))
    ax.set_xticklabels([f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins)-1)])
    ax.set_ylim((0, 1))
    fig.suptitle('Mean Absolute Error between confidence\nof GAN and BB prediction confidence')
    plt.tight_layout()
    # plt.legend()
    fig_path = os.sep.join([experiment_dir, f'mean_absolute_error_of_confidences_class_{order_of_classes[proba_idx_to_measure]}.png'])
    plt.savefig(fig_path)



def train_gan_with_twist_and_generate_statistics(random_forest_model, input_size, columns_size, column_idx_to_scaler, column_idx_to_ohe, num_samples,
                                                 experiment_dir, logger, df_real_not_normalized, labels_to_num_dict):

    df_columns = df_real_not_normalized.iloc[:, :-1].columns.tolist()
    # Gather numeric and categorical columns into a list
    numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(df_real_not_normalized.iloc[:, :-1])

    if GANBBMODEL_OBJECTIVE_FUNCTION == 'binary_crossentropy':
        gan_bb_model = GANBBModelBinaryCE(random_forest_model.model, input_size, columns_size)
    else:
        gan_bb_model = GANBBModel(random_forest_model.model, input_size, columns_size, generator_lr=GENERATOR_LR, discriminator_lr=CRITIC_LR, discriminator_dropout=CRITIC_DROPOUT)
    gan_sample_generator = GanWithTwistSampleGenerator(LATENT_NOISE_SIZE,
                                                       column_idx_to_scaler,
                                                       column_idx_to_ohe,
                                                       columns_size,
                                                       num_samples,
                                                       evaluation_mode=True)
    d_loss_epoch, g_loss_epoch, min_mse_score_for_fixed_latent_noise_and_confidence, \
    min_mse_score_for_random_latent_noise_and_confidence, samples, generated_samples, generated_confidence_scores, \
    score_for_fixed_latent_noise_and_confidence_hist, score_for_random_latent_noise_and_confidence_hist = gan_bb_model.train(
        num_samples,
        BATCH_SIZE,
        gan_sample_generator,
        N_EPOCHS,
        df_real_not_normalized.iloc[:, :-1],
        experiment_dir,
        logger)

    # table evaluation
    df_generated_samples = pd.DataFrame(data=np.array(samples), columns=df_columns)
    df_generated_samples[numeric_columns] = df_generated_samples[numeric_columns].apply(pd.to_numeric)

    # save fake dataframe
    df_generated_samples.to_csv(f"{experiment_dir}/df_generated_samples.csv", index=False)

    # generate profile report on the generated samples to detect mode collapse
    df_generated_samples_eda_output_file = f"{experiment_dir}/df_generated_samples.html"
    design_report = ProfileReport(df_generated_samples)
    design_report.to_file(output_file=df_generated_samples_eda_output_file)

    # line plots of loss
    plot_critic_generator_loss(list(range(1, len(d_loss_epoch) + 1)), d_loss_epoch, list(range(1, len(g_loss_epoch) + 1)), g_loss_epoch,
                               "critic loss", "generator loss", "epoch #", "loss", "Discriminator and Generator loss values per epoch", experiment_dir)

    # line plot of scores
    plot_score_metrics(list(range(1, len(score_for_fixed_latent_noise_and_confidence_hist) + 1)), score_for_fixed_latent_noise_and_confidence_hist,
                       list(range(1, len(score_for_random_latent_noise_and_confidence_hist) + 1)), score_for_random_latent_noise_and_confidence_hist, "fixed latent noise", "random latent noise", "epoch", "score", "MSE_pred_c", experiment_dir)

    logger.info("")
    logger.info("Best MSE score fixed latent noise and confidence scores: {}, random latent noise and random confidence scores: {}".format(
        min_mse_score_for_fixed_latent_noise_and_confidence,
        min_mse_score_for_random_latent_noise_and_confidence))
    logger.info("")

    # part 4
    # b - generate 1000 samples
    generator = load_model(f"{experiment_dir}/generator.h5")

    samples, generated_samples, generated_confidence_scores = gan_sample_generator.generate_samples(generator, random_latent_noise_and_confidence_scores=True)

    # extract random samples
    num_of_random_samples = NUM_OF_RANDOM_SAMPLES_PART2
    generated_samples_reduced = np.array(generated_samples)[:num_of_random_samples, :]
    generated_confidence_scores_reduced = generated_confidence_scores[:num_of_random_samples]
    samples_reduced = np.array(samples)[:num_of_random_samples, :]

    part_2_section_4_c(generated_samples_reduced, generated_confidence_scores_reduced, clf=random_forest_model, labels_to_num_dict=labels_to_num_dict,
                       experiment_dir=experiment_dir, proba_idx_to_measure=0)

    part_2_section_4_c(generated_samples_reduced, generated_confidence_scores_reduced, clf=random_forest_model,
                       labels_to_num_dict=labels_to_num_dict,
                       experiment_dir=experiment_dir, proba_idx_to_measure=1)


def classifier_evaluation(labels_to_num_dict: dict,
                          data_path: str,
                          target_column_name,
                          experiment_dir,
                          logger):
    logger.info(f"Train RandomForestClassifier model on {data_path} data and evaluate")
    model = SimpleCLFForEvaluation(labels_to_num_dict, data_path=data_path)
    score = model.train_and_score_model()
    logger.info(f'model score on real data: {score}')
    logger.info("")

    model_confidence_score_distribution(model, experiment_dir, logger, hue=target_column_name)

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

    if GAN_MODE == "gan_with_twist":
        experiment_name += "objective={}".format(GANBBMODEL_OBJECTIVE_FUNCTION)

    section = "section2" if GAN_MODE == "gan_with_twist" else "section1"
    experiment_dir = os.sep.join([DATASET, section, experiment_name])
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    log_filename = os.sep.join([experiment_dir, f'{experiment_name}.txt'])

    logger = log(".", log_filename)
    logger.info("#################################################################")

    # classifier evaluation
    random_forest_model = None
    if GAN_MODE == 'gan_with_twist':
        random_forest_model = SimpleCLFForEvaluation(labels_to_num_dict, data_path=dataset_path)
        random_forest_model.train_model()
        model_confidence_score_distribution(random_forest_model, experiment_dir, logger, hue=target_column_name)
        X_test, y_test = random_forest_model.X_test, random_forest_model.y_test
    else:
        X_test, y_test = classifier_evaluation(labels_to_num_dict, dataset_path, target_column_name, experiment_dir, logger)

    df_real_normalized = pd.concat([X, y], axis=1)
    df_real_not_normalized = pd.concat([df_real_not_normalized, y], axis=1)
    positive_negative_labels = [0, 1]

    if GAN_MODE == 'cgan':
        train_cgan_and_generate_statistics(ds, df_real_normalized, input_size, columns_size, num_classes,
                                           column_idx_to_scaler, column_idx_to_ohe,
                                           num_samples, X_test, y_test, num_positive_negative_classes, positive_negative_labels, experiment_dir, logger, df_real_not_normalized)
    elif GAN_MODE == 'cwgan':
        positive_negative_labels = [1, -1]
        train_cwgan_and_generate_statistics(X, y, df_real_normalized, input_size, columns_size, num_classes,
                                            column_idx_to_scaler, column_idx_to_ohe,
                                            num_samples, X_test, y_test, num_positive_negative_classes, positive_negative_labels, experiment_dir, logger, df_real_not_normalized)
    else:
        train_gan_with_twist_and_generate_statistics(random_forest_model, input_size, columns_size, column_idx_to_scaler,
                                                     column_idx_to_ohe, num_samples, experiment_dir, logger, df_real_not_normalized, labels_to_num_dict)


if __name__ == '__main__':
    if PERFORM_GRID_SERACH:
        global BATCH_SIZE
        global N_EPOCHS
        global GENERATOR_LR
        global CRITIC_LR
        global CRITIC_DROPOUT
        global GAN_MODE

        # run grid search
        gan_modes = [MODELS[0], MODELS[1]]
        batch_sizes = [16, 32]
        n_epochs = [200]
        generator_lr = [0.0005, 0.00005, 0.000005]
        critic_lr = [0.0005, 0.00005, 0.000005]
        critic_dropout = [0.2, 0.5]

        for gan_mode in gan_modes:
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
                                GAN_MODE = gan_mode

                                main()

    else:
        main()
