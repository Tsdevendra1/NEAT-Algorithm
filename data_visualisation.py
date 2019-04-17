import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import copy

from NEAT import NEAT
from genome_neural_network import GenomeNeuralNetwork
from neural_network import create_data
from data_storage import get_circle_data, get_spiral_data


def initialise_genome(genome_pickle_filepath):
    """
    Function to intialise a genome from a pickle file
    :param genome_pickle_filepath: File path to pickle
    :return: the intialised genome
    """
    infile = open(genome_pickle_filepath, 'rb')
    genome = pickle.load(infile)
    infile.close()
    return genome


def get_genome_predictions(genome, x_data, y_data):
    """
    Function to return predictions for a given genome
    :param genome: The genome class instance
    :param x_data:  The data to be predicted on
    :param y_data: The true labels for the data
    :return: the predictions for the given x_data
    """
    genome_nn = NEAT.create_genome_nn(genome=genome, x_data=x_data, y_data=y_data)
    return genome_nn.run_one_pass(input_data=x_data, return_prediction_only=True).round()


def plot_decision_boundary(genome, x_data):
    # UNCOMMENT IF USING XOR
    # x_values = np.linspace(np.min(x_data), np.max(x_data), 200).tolist()
    x_values = np.linspace(-4, 4, 200).tolist()
    x_data, y_data = create_data(n_generated=1)
    y_values = []

    random = copy.deepcopy(x_values)
    random.reverse()

    x1 = []
    x2 = []

    for x in x_values:
        for y in random:
            # x_data = np.array([[x, y, x ** 2, y ** 2, x * y, np.sin(x), np.sin(y)]])
            x_data = np.array([[x, y]])
            x1.append(x)
            x2.append(y)
            predictions = get_genome_predictions(genome=genome, x_data=x_data, y_data=y_data)
            y_values += predictions[0].tolist()
    for x in x_values:
        for y in random:
            # x_data = np.array([[y, x, y ** 2, x ** 2, y * x, np.sin(y), np.sin(x)]])
            x_data = np.array([[y, x]])
            x1.append(y)
            x2.append(x)
            predictions = get_genome_predictions(genome=genome, x_data=x_data, y_data=y_data)
            y_values += predictions[0].tolist()
    plt.scatter(x1, x2, color=create_label_colours(labels=np.array(y_values)))
    plt.title('Decisionary boundary for optimized genome')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    print(x_values)


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


def overlay_results():
    pass


def main():
    # DATA
    x_data, y_data = create_data(n_generated=200, add_noise=False)
    x_circle, y_circle = get_circle_data()
    x_spiral, y_spiral = get_spiral_data()

    # X1, X2 for all datasets
    feature_1_xor = x_data[:, 0]
    feature_2_xor = x_data[:, 1]
    feature_1_circle = x_circle[:, 0]
    feature_2_circle = x_circle[:, 1]
    feature_1_spiral = x_spiral[:, 0]
    feature_2_spiral = x_spiral[:, 1]

    plot_data = False
    show_decision_boundary = True

    # PLOT DATA
    if plot_data:
        # TODO: Add legends
        plt.scatter(feature_1_xor, feature_2_xor, color=create_label_colours(labels=y_data))
        plt.title('XOR Data')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
        plt.scatter(feature_1_circle, feature_2_circle, color=create_label_colours(labels=y_circle))
        plt.title('Circle Data')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
        plt.scatter(feature_1_spiral, feature_2_spiral, color=create_label_colours(labels=y_spiral))
        plt.title('Spiral Data')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()

    if show_decision_boundary:
        # Test genome accuracy
        genome = initialise_genome(genome_pickle_filepath='pickles/best_genome_pickle_circle_data_8')
        plot_decision_boundary(genome=genome, x_data=x_circle)


if __name__ == "__main__":
    main()
