from typing import List

import matplotlib.pyplot as plt


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
