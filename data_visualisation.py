import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D

from NEAT import NEAT
from genome_neural_network import GenomeNeuralNetwork
from neural_network import create_data
from data_storage import get_circle_data, get_spiral_data
from read_mat_files import get_shm_two_class_data


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


def get_genome_predictions(genome, x_data):
    """
    Function to return predictions for a given genome
    :param genome: The genome class instance
    :param x_data:  The data to be predicted on
    :param y_data: The true labels for the data
    :return: the predictions for the given x_data
    """
    # y_data isn't important but it's needed as a parameter
    _, y_data = create_data(n_generated=500)
    genome_nn = NEAT.create_genome_nn(genome=genome, x_data=x_data, y_data=y_data)
    return genome_nn.run_one_pass(input_data=x_data, return_prediction_only=True).round()


def plot_decision_boundary(genome, data_being_used):
    assert (data_being_used in {'circle_data', 'xor_data', 'spiral_data', 'shm_two_class'})

    number_of_data_points = 50
    if data_being_used == 'xor_data':
        x_values = np.linspace(0, 1, number_of_data_points).tolist()
    elif data_being_used == 'circle_data':
        x_values = np.linspace(-4, 4, number_of_data_points).tolist()
    elif data_being_used == 'shm_two_class':
        x_values = np.linspace(-29, 1, number_of_data_points).tolist()
        y_values = np.linspace(-34, 4, number_of_data_points).tolist()
        z_values = np.linspace(-31, 11, number_of_data_points).tolist()

    prediction_list = []
    if data_being_used != 'shm_two_class':
        x_values_reverse = copy.deepcopy(x_values)
        x_values_reverse.reverse()
        current_x = []
        current_y = []
        for x in x_values:
            for y in x_values_reverse:
                # x_data = np.array([[x, y, x ** 2, y ** 2, x * y, np.sin(x), np.sin(y)]])
                x_data = np.array([[x, y]])
                current_x.append(x)
                current_y.append(y)
                predictions = get_genome_predictions(genome=genome, x_data=x_data)
                prediction_list += predictions[0].tolist()
        for x in x_values:
            for y in x_values_reverse:
                # x_data = np.array([[y, x, y ** 2, x ** 2, y * x, np.sin(y), np.sin(x)]])
                x_data = np.array([[y, x]])
                # This is correct, should be reverse to previous loop
                current_x.append(y)
                current_y.append(x)
                predictions = get_genome_predictions(genome=genome, x_data=x_data)
                prediction_list += predictions[0].tolist()
        plt.scatter(current_x, current_y, color=create_label_colours(labels=np.array(prediction_list)))
        plt.title('Decisionary boundary for optimized genome')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()

    else:
        # REMEMBER WHEN PLOTTING SHM DATA NEED TO MINUS AND DIVIDE BY 10
        current_x = []
        current_y = []
        current_z = []

        x_values_reverse = copy.deepcopy(x_values)
        x_values_reverse.reverse()

        y_values_reverse = copy.deepcopy(y_values)
        y_values_reverse.reverse()

        z_values_reverse = copy.deepcopy(z_values)
        z_values_reverse.reverse()
        for x in x_values:
            for y in y_values:
                for z in z_values:
                    print(x, y, z)
                    # x_data = np.array([[x, y, x ** 2, y ** 2, x * y, np.sin(x), np.sin(y)]])
                    x_data = np.array([[x, y, z]])
                    current_x.append(x)
                    current_y.append(y)
                    current_z.append(z)
                    predictions = get_genome_predictions(genome=genome, x_data=x_data)
                    prediction_list += predictions[0].tolist()
        for x in x_values_reverse:
            for y in y_values:
                for z in z_values:
                    print(x, y, z)
                    # x_data = np.array([[x, y, x ** 2, y ** 2, x * y, np.sin(x), np.sin(y)]])
                    x_data = np.array([[x, y, z]])
                    current_x.append(x)
                    current_y.append(y)
                    current_z.append(z)
                    predictions = get_genome_predictions(genome=genome, x_data=x_data)
                    prediction_list += predictions[0].tolist()
        for x in x_values:
            for y in y_values_reverse:
                for z in z_values:
                    print(x, y, z)
                    # x_data = np.array([[x, y, x ** 2, y ** 2, x * y, np.sin(x), np.sin(y)]])
                    x_data = np.array([[x, y, z]])
                    current_x.append(x)
                    current_y.append(y)
                    current_z.append(z)
                    predictions = get_genome_predictions(genome=genome, x_data=x_data)
                    prediction_list += predictions[0].tolist()
        for x in x_values:
            for y in y_values:
                for z in z_values_reverse:
                    # x_data = np.array([[x, y, x ** 2, y ** 2, x * y, np.sin(x), np.sin(y)]])
                    x_data = np.array([[x, y, z]])
                    current_x.append(x)
                    current_y.append(y)
                    current_z.append(z)
                    predictions = get_genome_predictions(genome=genome, x_data=x_data)
                    prediction_list += predictions[0].tolist()

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(current_x, current_y, current_z, color=create_label_colours(np.array(prediction_list)))
        ax.view_init(-140, 30)
        plt.show()


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


def plot_shm_data(rotation_angle):
    x_data, y_data = get_shm_two_class_data(normalise_x=False)

    x_vals = x_data[:, 0].tolist()
    y_vals = x_data[:, 1].tolist()
    z_vals = x_data[:, 2].tolist()

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    z_min, z_max = min(z_vals), max(z_vals)

    labels = create_label_colours(labels=y_data)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_vals, y_vals, z_vals, color=labels)
    ax.view_init(-140, rotation_angle)
    plt.show()


def plot_generation_graph(*args, same_axis=None, generation_information, y_label=None, title):
    """"
    Generic function to plot data
    :param title: String for the title
    :param y_label: String for the y label
    :param same_axis: Defines whether two or more datasets should be plotted on the same y axis
    """
    # Plus one because of how the range function works
    generations_to_go_through = list(range(1, max(generation_information) + 1))

    if len(args) > 1:

        # Can't plot more than two items on a 2d plot
        assert (len(args) == 2)
        assert (same_axis is not None)
        if same_axis:
            # Need a common y_label
            assert (y_label is not None)

        y_data_list = []
        y_labels = []
        for information in args:
            information_type = information[0]
            information_plot_type = information[1]
            if not same_axis:
                y_label = information[2]
                y_labels.append(y_label)

            y_data = []
            for generation in generations_to_go_through:
                y_data.append(generation_information[generation][information_type])
            if information_plot_type == 'line' and same_axis:
                plt.plot(generations_to_go_through, y_data)
            elif information_plot_type == 'bar' and same_axis:
                plt.bar(generations_to_go_through, y_data)
            y_data_list.append(y_data)

        if not same_axis:
            plt.plot(generations_to_go_through, y_data_list[0], color='r')
            plt.ylabel(y_labels[0])
            axes2 = plt.twinx()
            axes2.plot(generations_to_go_through, y_data_list[1], color='g')
            axes2.set_ylabel(y_labels[1])
        else:
            plt.ylabel(y_label)
        plt.xticks(generations_to_go_through)
        plt.xlabel('Generation')
        plt.title(title)
        plt.show()

    else:
        y_data = []
        information = args[0]
        information_type = information[0]
        information_plot_type = information[1]
        for generation in generations_to_go_through:
            y_data.append(generation_information[generation][information_type])
        if information_plot_type == 'line':
            plt.plot(generations_to_go_through, y_data)
        elif information_plot_type == 'bar':
            plt.plot(generations_to_go_through, y_data)
        plt.xticks(generations_to_go_through)
        plt.xlabel('Generation')
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()


def visualise_generation_tracker(filepath_to_genome):
    infile = open(filepath_to_genome, 'rb')
    generation_tracker_instance = pickle.load(infile)
    generation_information_dict = generation_tracker_instance.generation_information

    # If more than one information type is specified, MUST define the same_axis variable
    plot_generation_graph(('best_all_time_genome_accuracy', 'line', 'Best Genome Accuracy (%)'),
                          ('best_all_time_genome_f1_score', 'line', 'Best Genome F1 score'),
                          same_axis=False,
                          generation_information=generation_information_dict,
                          title='Best All Time Genome Accuracy through generations')

    plot_generation_graph(('best_all_time_genome_accuracy', 'line'),
                          generation_information=generation_information_dict, y_label='Best Genome Accuracy (%)',
                          title='Best All Time Genome Accuracy through generations')
    infile.close()


def main():
    # plot_shm_data(rotation_angle=50)

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
    show_decision_boundary = False

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
        genome = initialise_genome(genome_pickle_filepath='pickles/best_genome_pickle_shm_two_class_618056')
        plot_decision_boundary(genome=genome, data_being_used='shm_two_class')

    visualise_generation_tracker(filepath_to_genome='algorithm_runs/xor_full/run_1/generation_tracker')
    #
    # plt.figure()
    # N = 5
    # menMeans = (20, 35, 30, 35, 27)
    # menStd = (2, 3, 4, 1, 2)
    # width = 0.35  # the width of the bars
    # womenMeans = (25, 32, 34, 20, 25)
    # womenStd = (3, 5, 2, 3, 3)
    # ind = np.arange(N)
    # plt.ylim(0.0, 65.0)
    # plt.bar(ind, menMeans, width, color='r', yerr=menStd, label='Men means')
    # plt.bar(ind + width, womenMeans, width, color='y', yerr=womenStd, label='Women means')
    # # plt.plot(ind + width, womenMeans, color='k', label='Sine')
    # plt.ylabel('Bar plot')
    #
    # x = np.linspace(0, N)
    # y = np.sin(x)
    # axes2 = plt.twinx()
    # # axes2.plot(ind+width, womenMeans, color='k', label='Sine')
    # axes2.plot(x, y, color='k', label='Sine')
    # # axes2.set_ylim(-1, 1)
    # # axes2.set_ylabel('Line plot')
    #
    # plt.show()


if __name__ == "__main__":
    main()
