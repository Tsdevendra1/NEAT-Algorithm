import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from neural_network import create_data
from data_storage import get_circle_data, get_spiral_data


def create_label_colours(labels):
    """
    Function turns the binary classification classes into two seperate colours e.g. 1, 0 => 'green', 'red'
    :param labels: numpy array of labels in shape (n, 1)
    :return:
    """
    main_features = np.unique(labels)
    if main_features.shape[0] != 2:
        raise ValueError('There can only be two class labels')
    try:
        if labels.shape[1] != 1:
            raise ValueError('Labels must be in shape (num_examples, 1)')
        labels_list = labels[:, 0]
    except IndexError:
        labels_list = labels[:, ]
    coloured_labels = ['red' if label == main_features[0] else 'green' for label in labels_list]
    return coloured_labels


def main():
    # DATA
    x_data, y_data = create_data(n_generated=200, add_noise=True)
    x_circle, y_circle = get_circle_data()
    x_spiral, y_spiral = get_spiral_data()

    # X1, X2 for all datasets
    feature_1_xor = x_data[:, 0]
    feature_2_xor = x_data[:, 1]
    feature_1_circle = x_circle[:, 0]
    feature_2_circle = x_circle[:, 1]
    feature_1_spiral = x_spiral[:, 0]
    feature_2_spiral = x_spiral[:, 1]

    # PLOT DATA
    plt.scatter(feature_1_xor, feature_2_xor, color=create_label_colours(labels=y_data))
    plt.show()
    plt.scatter(feature_1_circle, feature_2_circle, color=create_label_colours(labels=y_circle))
    plt.show()
    plt.scatter(feature_1_spiral, feature_2_spiral, color=create_label_colours(labels=y_spiral))
    plt.show()


if __name__ == "__main__":
    main()
