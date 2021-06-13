import logging
from typing import List, Tuple

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from keras.models import load_model

from global_vars import SEED
from preprocessing_utils import gather_numeric_and_categorical_columns


def log(path, file):
    # check if the file exist
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = "%(message)s"
    file_logging_format = "%(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger(file)

    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


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


def plot_loss_history(d1_hist, d2_hist, g_hist, experiment_dir):
    # plot loss history
    plt.plot(d1_hist, label='crit_real')
    plt.plot(d2_hist, label='crit_fake')
    plt.plot(g_hist, label='gen')
    plt.legend()
    plt.savefig(f'{experiment_dir}/loss_plot.png')
    plt.close()


def plot_accuracy_history(d1_acc, d2_acc, experiment_dir):
    # plot accuracy
    plt.plot(d1_acc, label='crit_real')
    plt.plot(d2_acc, label='crit_fake')
    plt.legend()
    plt.savefig(f'{experiment_dir}/accuracy_plot.png')
    plt.close()


def tsne(df: pd.DataFrame, non_numeric_columns: list, hue: str, filename: str = '', save_figure: bool = False):
    df_copy = df.copy()
    # Drop the non-numerical columns from df
    df_numeric = df_copy.drop(non_numeric_columns, axis=1)

    # Create a t-SNE model with learning rate 50
    m = TSNE(learning_rate=50, random_state=SEED, n_jobs=-1)

    # Fit and transform the t-SNE model on the numeric dataset
    tsne_features = m.fit_transform(df_numeric)

    df_copy = df_copy.join(pd.DataFrame(tsne_features, columns=['x', 'y']))

    sns.scatterplot(x="x", y="y", hue=hue, data=df_copy)
    if save_figure:
        plt.savefig(filename + ".png")
        plt.clf()
    else:
        plt.show()


def model_confidence_score_distribution(model):
    confidence_score_distribution_arr = model.model_confidence_score_distribution()

    y_test_copy = model.y_test.copy()
    y_test_copy.reset_index(drop=True, inplace=True)
    sns.displot(pd.concat([pd.DataFrame(confidence_score_distribution_arr, columns=["prediction"]), y_test_copy], axis=1), x="prediction", hue="class")
    plt.title("Confidence score distribution histogram with 'auto' bins")
    plt.show()

    df_describe = pd.DataFrame(confidence_score_distribution_arr)
    print("Confidence score distribution statistics: {}".format(df_describe.describe()))

    return confidence_score_distribution_arr


def evaluate_machine_learning_efficacy(generated_samples, labels, X_test, y_test):
    model = RandomForestClassifier(random_state=SEED)
    model.fit(generated_samples, labels)

    return model.score(X_test, y_test)


def evaluate_using_tsne(samples, labels, df_columns, index, dir):
    df = pd.DataFrame(data=np.concatenate((np.array(samples), labels.reshape(-1, 1)), axis=1), columns=df_columns + ['class'])
    numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(df)
    tsne(df, categorical_columns, hue='class', filename=f'{dir}/{index}_tsne', save_figure=True)


def draw_boxplot(samples:np.array, generated_samples: np.array, path_to_save_fig: str):
    def draw_plot(ax, data, offset,edge_color, fill_color, label: str):
        pos = np.arange(data.shape[1])+offset
        bp = ax.boxplot(data, positions= pos, widths=0.2, patch_artist=True)#, label=label)
        for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color)
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)
        return bp
    fig, ax = plt.subplots()
    bp1 = draw_plot(ax, samples, -0.1, "red", "white", label='Real')
    bp2 = draw_plot(ax, generated_samples, +0.1, "blue", "white", label='Generated')
    plt.xticks(ticks=np.arange(samples.shape[1]), labels=[f'F{i+1}' for i in np.arange(samples.shape[1])])
    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Real', 'Generated'], loc='upper right')
    plt.tight_layout()
    plt.savefig(path_to_save_fig, dpi=200)


# def real_to_generated_distance(real: np.array, generated: np.asarray):


def generate_and_draw_boxplots(experiment_dir, gan_sample_generator, df_real, num_of_samples):
    generator = load_model(f"{experiment_dir}/generator.h5")

    samples, generated_samples, labels_input = gan_sample_generator.generate_samples(generator,
                                                                                     random_latent_noise=True)
    # extract N random samples
    generated_samples_reduced = np.array(generated_samples)[:num_of_samples, :]

    path_to_box_plot = os.sep.join([experiment_dir, 'boxplot_save.png'])
    draw_boxplot(samples=df_real.values[:num_of_samples, :-1],
                 generated_samples=np.asarray(generated_samples_reduced),
                 path_to_save_fig=path_to_box_plot)


class GanSampleGenerator:
    def __init__(self,
                 latent_noise_size: int,
                 column_idx_to_scaler: dict,
                 column_idx_to_ohe: dict,
                 columns_size: List[int],
                 num_samples: int = 1,
                 is_label_conditional: bool = True,
                 num_positive_negative_classes: Tuple[int, int] = None,
                 evaluation_mode: bool = False,
                 positive_negative_labels: List[int] = None):
        if positive_negative_labels is None:
            self.positive_negative_labels = [0, 1]
        else:
            self.positive_negative_labels = positive_negative_labels

        self.is_label_conditional = is_label_conditional
        self.columns_size = columns_size
        self.evaluation_mode = evaluation_mode
        self.num_positive_negative_classes = num_positive_negative_classes
        self.num_samples = num_samples
        self.column_idx_to_ohe = column_idx_to_ohe
        self.column_idx_to_scaler = column_idx_to_scaler
        self.latent_noise_size = latent_noise_size
        self.z_input = None

        if self.evaluation_mode:
            # sample random noise latent vectors
            self.z_input = tf.random.truncated_normal(shape=[self.latent_noise_size * self.num_samples]).numpy()
            # reshape into a batch of inputs for the network
            self.z_input = self.z_input.reshape(self.num_samples, self.latent_noise_size)

        self.labels_input = None
        if self.is_label_conditional and self.num_positive_negative_classes is not None:
            # sample labels according to given parameter
            self.positive_labels = np.full((self.num_positive_negative_classes[1],), self.positive_negative_labels[1])
            self.negative_labels = np.full((self.num_positive_negative_classes[0],), self.positive_negative_labels[0])
            self.labels_input = np.concatenate((self.positive_labels, self.negative_labels))
            np.random.shuffle(self.labels_input)

    def generate_samples(self, generator, random_latent_noise: bool = False):
        if not random_latent_noise and self.evaluation_mode:
            z_input = self.z_input
        else:
            # sample random noise latent vectors
            z_input = tf.random.truncated_normal(shape=[self.latent_noise_size * self.num_samples]).numpy()
            # reshape into a batch of inputs for the network
            z_input = z_input.reshape(self.num_samples, self.latent_noise_size)

        if self.labels_input is not None:
            labels = self.labels_input
        else:
            # sample random labels
            labels = np.random.choice(self.positive_negative_labels, self.num_samples)

        # generate samples using generator model
        if self.is_label_conditional:
            generated_samples = generator.predict([z_input, labels])
        else:
            generated_samples = generator.predict(z_input)
        generated_samples = generated_samples.tolist()

        # convert raw generated samples' representation into original format
        samples = []
        for generated_sample in generated_samples:
            sample = []
            column_idx = 0
            column_size_idx = len(self.column_idx_to_scaler)

            for sample_col_value in generated_sample:
                if column_idx in self.column_idx_to_scaler.keys():  # inverse transform min-max scaler
                    sample.append(self.column_idx_to_scaler[column_idx].inverse_transform(np.array([[sample_col_value]]))[0][0])
                else:  # inverse transform one-hot-encoding format
                    if column_idx not in self.column_idx_to_ohe.keys():
                        column_idx += 1
                        continue

                    categorical_softmax_representation = generated_sample[column_idx:column_idx + self.columns_size[column_size_idx]]
                    # find index with the max value and generate one-hot-encoding representation
                    max_index = np.argmax(np.array(categorical_softmax_representation))
                    categorical_ohe_representation = [0] * self.columns_size[column_size_idx]
                    categorical_ohe_representation[max_index] = 1
                    categorical_value = self.column_idx_to_ohe[column_idx].inverse_transform([categorical_ohe_representation])[0][0]

                    sample.append(categorical_value)
                    column_size_idx += 1

                column_idx += 1

            samples.append(sample)

        return samples, generated_samples, labels
