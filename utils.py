from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from global_vars import SEED


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