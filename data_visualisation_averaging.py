import numpy as np
import pandas as pd
import seaborn as sn
import pickle
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D

from NEAT import NEAT
from genome_neural_network import GenomeNeuralNetwork
from neural_network import create_data
from data_storage import get_circle_data, get_spiral_data
from read_mat_files import get_shm_two_class_data
import os


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


def get_genome_predictions(genome, x_data, round_values=True):
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
    predictions = genome_nn.run_one_pass(input_data=x_data, return_prediction_only=True)
    if round_values:
        predictions = predictions.round()
    return predictions


def plot_decision_boundary(experiments_path, data_being_used):
    number_of_data_points = 110
    assert (data_being_used in {'circle_data', 'xor_data', 'spiral_data', 'shm_two_class'})

    if data_being_used == 'xor_data':
        x_values = np.linspace(0, 1, number_of_data_points).tolist()
    elif data_being_used == 'circle_data':
        x_values = np.linspace(-4, 4, number_of_data_points).tolist()
    elif data_being_used == 'shm_two_class':
        x_values = np.linspace(-29, 1, number_of_data_points).tolist()
        y_values = np.linspace(-34, 4, number_of_data_points).tolist()
        z_values = np.linspace(-31, 11, number_of_data_points).tolist()

    list_of_all_genome_predictions = []
    # We go through all the experiments conducted
    for _, dirnames, filenames in os.walk(experiments_path):
        for directory in dirnames:
            genome = initialise_genome(
                genome_pickle_filepath='{}/{}/best_genome_pickle'.format(experiments_path, directory))

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
                        predictions = get_genome_predictions(genome=genome, x_data=x_data, round_values=False)
                        # prediction_value = predictions[0][0]
                        # if (x < 0.5 and y < 0.5 and prediction_value < 0.5):
                        #     ewqeq = 3
                        # if (x > 0.5 and y > 0.5 and prediction_value < 0.5):
                        #     ewqeq = 3
                        # if (x > 0.5 and y < 0.5 and prediction_value > 0.5):
                        #     ewqeq = 3
                        # if (x < 0.5 and y > 0.5 and prediction_value > 0.5):
                        #     ewqeq = 3
                        prediction_list += predictions[0].tolist()
                # for x in x_values:
                #     for y in x_values_reverse:
                #         # x_data = np.array([[y, x, y ** 2, x ** 2, y * x, np.sin(y), np.sin(x)]])
                #         x_data = np.array([[y, x]])
                #         # This is correct, should be reverse to previous loop
                #         current_x.append(y)
                #         current_y.append(x)
                #         predictions = get_genome_predictions(genome=genome, x_data=x_data, round_values=False)
                #         # prediction_value = predictions[0][0]
                #         # if (x < 0.5 and y < 0.5 and prediction_value < 0.5):
                #         #     ewqeq = 3
                #         # if (x > 0.5 and y > 0.5 and prediction_value < 0.5):
                #         #     ewqeq = 3
                #         # if (x > 0.5 and y < 0.5 and prediction_value > 0.5):
                #         #     ewqeq = 3
                #         # if (x < 0.5 and y > 0.5 and prediction_value > 0.5):
                #         #     ewqeq = 3
                #         prediction_list += predictions[0].tolist()
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
            list_of_all_genome_predictions.append(prediction_list)

        averaged_predictions = []
        for index in range(len(list_of_all_genome_predictions[0])):
            y_predictions_current = []
            for genome_prediction in list_of_all_genome_predictions:
                y_predictions_current.append(genome_prediction[index])
            averaged_predictions.append(np.mean(y_predictions_current))

        rounded_average_predictions = np.array(averaged_predictions).round()
        if data_being_used != 'shm_two_class':
            plt.scatter(current_x, current_y, color=create_label_colours(labels=rounded_average_predictions))
            plt.title('Decisionary boundary for optimized genome')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.show()

            fig, ax = plt.subplots()
            label_colours = create_label_colours(labels=rounded_average_predictions)
            x1_reds = []
            x2_reds = []
            x1_greens = []
            x2_greens = []
            for index in range(len(label_colours)):
                if label_colours[index] == 'green':
                    x1_greens.append(current_x[index])
                    x2_greens.append(current_y[index])
                else:
                    x1_reds.append(current_x[index])
                    x2_reds.append(current_y[index])

            ax.scatter(x1_greens, x2_greens, c='green', label='Class 1',
                       alpha=1, edgecolors='none')
            ax.scatter(x1_reds, x2_reds, c='red', label='Class 0',
                       alpha=1, edgecolors='none')
            ax.legend(loc='upper right')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.show()


        else:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(current_x, current_y, current_z, color=create_label_colours(np.array(prediction_list)))
            ax.view_init(-140, 30)
            plt.show()
        break


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


def plot_shm_data(rotation_angle, elevation, experiments_path):
    x_list = []
    y_list = []
    z_list = []
    predictions_list = []
    x_test_data = None
    y_test_data = None
    for directory in os.listdir(experiments_path):
        try:
            infile = open('{}/{}/NEAT_instance'.format(experiments_path, directory), 'rb')
            neat_instance = pickle.load(infile)
            if x_test_data is None:
                x_test_data = neat_instance.x_test
                y_test_data = neat_instance.y_test
            prediction = get_genome_predictions(genome=neat_instance.best_all_time_genome, x_data=x_test_data,
                                                round_values=False)
            predictions_list.append(prediction)
            infile.close()
        except:
            pass

    avg_prediction = []
    for i in range(len(predictions_list[0])):
        avg_tracker = []
        for run in predictions_list:
            avg_tracker.append(run[i])
        avg_prediction.append(np.mean(avg_tracker))

    avg_prediction = np.array(list(map(lambda x: int(x), np.array(avg_prediction).round().tolist())))
    avg_prediction.shape = (avg_prediction.shape[0], 1)

    x_test_data = x_test_data / -1
    x_test_data = x_test_data * 100
    x_data, y_data = get_shm_two_class_data(normalise_x=False)
    x_vals = x_test_data[:, 0].tolist()
    y_vals = x_test_data[:, 1].tolist()
    z_vals = x_test_data[:, 2].tolist()

    labels = create_label_colours(labels=avg_prediction)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_vals, y_vals, z_vals, color=labels)
    ax.view_init(elevation, rotation_angle)
    plt.show()

    fig, ax = plt.subplots()
    fig = plt.figure()
    ax = Axes3D(fig)
    x1_reds = []
    x2_reds = []
    x3_reds = []
    x1_greens = []
    x2_greens = []
    x3_greens = []
    for index in range(len(labels)):
        if labels[index] == 'green':
            x1_greens.append(x_vals[index])
            x2_greens.append(y_vals[index])
            x3_greens.append(z_vals[index])
        else:
            x1_reds.append(x_vals[index])
            x2_reds.append(y_vals[index])
            x3_reds.append(z_vals[index])

    ax.scatter(x1_greens, x2_greens, x3_greens, c='green', label='Undamaged',
               )
    ax.scatter(x1_reds, x2_reds, x3_reds, c='red', label='Damaged',
               )
    ax.legend(loc='upper right')
    ax.view_init(elevation, rotation_angle)
    plt.show()

    avg_prediction.shape = (avg_prediction.shape[0],)
    y_test_data.shape = (y_test_data.shape[0],)
    data = {'y_Predicted': avg_prediction.tolist(),
            'y_Actual': y_test_data.tolist(),
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    sn.heatmap(confusion_matrix, annot=True, xticklabels=['Undamaged', 'Damaged'], yticklabels=['Undamaged', 'Damaged'])
    plt.show()


def plot_generation_graph(*args, same_axis=None, generation_information, y_label=None, title):
    """"
    Generic function to plot data
    :param title: String for the title
    :param y_label: String for the y label
    :param same_axis: Defines whether two or more datasets should be plotted on the same y axis
    """
    # Plus one because of how the range function works
    generations_to_go_through = list(range(1, min(list(map(lambda x: max(x), generation_information))) + 1))
    # generations_to_go_through = list(range(1, max(generation_information[0]) + 1))

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
            if same_axis:
                label = information[2]
            if not same_axis:
                y_label = information[2]
                y_labels.append(y_label)

            y_data = []
            for generation in generations_to_go_through:
                y_avg_list = []
                for run in generation_information:
                    y_avg_list.append(run[generation][information_type])
                avg_value = np.mean(y_avg_list)
                y_data.append(avg_value)
            if information_plot_type == 'line' and same_axis:
                plt.plot(generations_to_go_through, y_data, label=label)
            elif information_plot_type == 'bar' and same_axis:
                plt.bar(generations_to_go_through, y_data, label=label)
            y_data_list.append(y_data)

        if not same_axis:
            plt.plot(generations_to_go_through, y_data_list[0], color='r')
            plt.ylabel(y_labels[0])
            axes2 = plt.twinx()
            axes2.plot(generations_to_go_through, y_data_list[1], color='g')
            axes2.set_ylabel(y_labels[1])
        else:
            plt.ylabel(y_label)
            plt.legend(loc='upper right')
        plt.xticks(generations_to_go_through)
        plt.xlabel('Generation')
        if title:
            plt.title(title)
        plt.show()

    else:
        y_data = []
        information = args[0]
        information_type = information[0]
        information_plot_type = information[1]
        for generation in generations_to_go_through:
            y_avg_list = []
            for run in generation_information:
                y_avg_list.append(run[generation][information_type])
            y_data.append(np.mean(y_avg_list))
        if information_plot_type == 'line':
            plt.plot(generations_to_go_through, y_data)
        elif information_plot_type == 'bar':
            plt.plot(generations_to_go_through, y_data)
        plt.xticks(generations_to_go_through)
        plt.xlabel('Generation')
        plt.ylabel(y_label)
        if title:
            plt.title(title)
        plt.show()


def visualise_generation_tracker(experiments_path):
    generation_information_list = []

    for directory in os.listdir(experiments_path):
        infile = open('{}/{}/generation_tracker'.format(experiments_path, directory), 'rb')
        generation_tracker_instance = pickle.load(infile)
        generation_information_list.append(generation_tracker_instance.generation_information)
        infile.close()

    # If more than one information type is specified, MUST define the same_axis variable
    plot_generation_graph(('best_all_time_genome_fitness', 'line', 'Best All Time Fitness'),
                          ('average_population_fitness', 'line', 'Population Average Fitness'),
                          same_axis=True,
                          y_label='Fitness',
                          generation_information=generation_information_list,
                          title=None)

    plot_generation_graph(('best_all_time_genome_accuracy', 'line'),
                          generation_information=generation_information_list, y_label='Best Genome Accuracy (%)',
                          title='Best All Time Genome Accuracy through generations')
    # plot_generation_graph(('best_all_time_genome_accuracy', 'line', 'Best Genome Accuracy (%)'),
    #                       ('best_all_time_genome_f1_score', 'line', 'Best Genome F1 score'),
    #                       same_axis=False,
    #                       generation_information=generation_information_list,
    #                       title='Best All Time Genome Accuracy through generations')
    #
    # plot_generation_graph(('best_all_time_genome_accuracy', 'line'),
    #                       generation_information=generation_information_list, y_label='Best Genome Accuracy (%)',
    #                       title='Best All Time Genome Accuracy through generations')


def plot_population_complexity(experiments_path):
    node_count_list = []
    connection_count_list = []
    best_connection_count_list = []
    best_node_count_list = []
    for directory in os.listdir(experiments_path):
        try:
            infile = open('{}/{}/NEAT_instance'.format(experiments_path, directory), 'rb')
            neat_instance = pickle.load(infile)
            connection_count = []
            node_count = []
            for population_member in neat_instance.population.values():
                node_count.append(len(population_member.nodes))
                connection_count.append(len(population_member.connections))
            node_count_list.append(node_count)
            connection_count_list.append(connection_count)
            best_connection_count_list.append(len(neat_instance.best_all_time_genome.connections))
            best_node_count_list.append(len(neat_instance.best_all_time_genome.nodes))
            infile.close()
        except:
            pass
    min_connection_list_length = None
    min_node_list_length = None
    for connection_list, node_list in zip(connection_count_list, node_count_list):
        connection_list.sort()
        node_list.sort()
        if min_connection_list_length is None or len(connection_list) < min_connection_list_length:
            min_connection_list_length = len(connection_list)
        if min_node_list_length is None or len(node_list) < min_node_list_length:
            min_node_list_length = len(node_list)

    avg_connection_count = []
    avg_node_count = []

    for index in range(min_connection_list_length):
        connections_count_keeper = []
        for connection_list in connection_count_list:
            connections_count_keeper.append(connection_list[index])
        avg_connection_count.append(np.mean(connections_count_keeper))

    for index in range(min_node_list_length):
        node_count_keeper = []
        for node_list in node_count_list:
            node_count_keeper.append(node_list[index])
        avg_node_count.append(np.mean(node_count_keeper))

    x_data = [(number + 1) for number in range(min_connection_list_length)]
    test = [np.mean(best_connection_count_list) for i in range(len(x_data))]

    plt.bar(x_data, avg_connection_count, label='Average number of connections')
    plt.xticks(x_data)
    plt.plot(x_data, avg_node_count, color='black', label='Average number of nodes', linestyle='--')
    plt.plot(x_data, test, color='r', label='Best genome connections')
    test = [np.mean(best_node_count_list) for i in range(len(x_data))]
    plt.plot(x_data, test, color='r', linestyle='--', label='Best genome nodes')
    # axes2 = plt.twinx()
    # axes2.plot(x_data, test, color='r')
    # axes2.plot(x_data, node_count, color='r')

    plt.xlabel('Individual')
    plt.ylabel('Complexity')
    plt.legend()
    # plt.title('Test title')
    plt.show()

    test = [np.mean(best_node_count_list) for i in range(len(x_data))]

    plt.bar(x_data, avg_node_count)
    plt.xticks(x_data)
    plt.xlabel('Individual')
    plt.ylabel('Test label')
    plt.title('Test title')
    plt.plot(x_data, test, color='r')

    plt.xlabel('Individual')
    plt.ylabel('Test label')
    plt.title('Test title')
    plt.show()

    # best_genome = neat_instance.best_all_time_genome
    #
    # print(len(best_genome.connections))
    # print(len(best_genome.nodes))
    #
    # infile.close()


def create_confusion_matrix(experiments_path, x_data, y_data):
    all_predictions_list = []
    for directory in os.listdir(experiments_path):
        genome = initialise_genome(
            genome_pickle_filepath='{}/{}/best_genome_pickle'.format(experiments_path, directory))
        predictions = get_genome_predictions(genome=genome, x_data=x_data, round_values=False)
        all_predictions_list.append(predictions)

    averaged_predictions = []
    for index in range(len(y_data)):
        avg_keeper = []
        for run in all_predictions_list:
            avg_keeper.append(run[index])
        averaged_predictions.append(np.mean(avg_keeper))

    averaged_predictions = np.array(averaged_predictions).round().tolist()
    y_data = y_data[:, 0].tolist()
    data = {'y_Predicted': averaged_predictions,
            'y_Actual': y_data,
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    sn.heatmap(confusion_matrix, annot=True)
    plt.show()


def plot_model_complexity_during_evolution(experiments_path):
    generation_information_list = []

    for directory in os.listdir(experiments_path):
        infile = open('{}/{}/generation_tracker'.format(experiments_path, directory), 'rb')
        generation_tracker_instance = pickle.load(infile)
        generation_information_list.append(generation_tracker_instance.generation_information)
        infile.close()

    complexity_tracker_nodes = {}
    complexity_tracker_connections = {}
    min_num_gens = None
    for generation_dict in generation_information_list:
        if min_num_gens is None or max(generation_dict) < min_num_gens:
            min_num_gens = max(generation_dict)
    for generation in range(1, min_num_gens + 1):
        avg_tracker_connections = []
        avg_tracker_nodes = []
        for generation_dict in generation_information_list:
            avg_tracker_connections.append(generation_dict[generation]['mean_number_connections_enabled'])
            avg_tracker_nodes.append(generation_dict[generation]['mean_number_nodes_enabled'])

        complexity_tracker_connections[generation] = np.mean(avg_tracker_connections)
        complexity_tracker_nodes[generation] = np.mean(avg_tracker_nodes)

    print(generation_information_list)
    plt.plot([1, 2, 3, 4, 5, 6, 7], list(complexity_tracker_connections.values()))
    plt.show()


def calculate_genome_fitness(neat_instance, genome):
    genome_nn = neat_instance.create_genome_nn(genome=genome, x_data=neat_instance.x_train,
                                               y_data=neat_instance.y_train,
                                               algorithm_running=neat_instance.algorithm_running)

    print('OPTIMISING')
    genome_nn.optimise(print_epoch=False)
    # We use genome_nn.x_train instead of self.x_train because the genome_nn might have deleted a row if there
    # is no connection to one of the sources
    cost = genome_nn.run_one_pass(input_data=genome_nn.x_train, labels=neat_instance.y_train, return_cost_only=True)
    print('OPTIMISING FINISH')

    # The fitness is the negative of the cost. Because less cost = greater fitness
    genome.fitness = -cost

    return -cost


def get_avg_table_values(experiments_path):
    fitness_list = []
    f1_score_list = []
    accuracy_list = []
    best_accuracy_list = []
    best_f1_list = []
    best_fitness_list = []
    for directory in os.listdir(experiments_path):
        try:
            infile = open('{}/{}/NEAT_instance'.format(experiments_path, directory), 'rb')
            neat_instance = pickle.load(infile)
            for genome in neat_instance.population.values():
                f_score = neat_instance.calculate_f_statistic(genome=genome, x_test_data=neat_instance.x_test,
                                                              y_test_data=neat_instance.y_test)
                accuracy = neat_instance.calculate_accuracy(genome=genome, x_test_data=neat_instance.x_test,
                                                            y_test_data=neat_instance.y_test)
                accuracy_list.append(accuracy)
                f1_score_list.append(f_score)
                # fitness_value = genome.fitness
                # if not fitness_value:
                #     fitness_value = calculate_genome_fitness(neat_instance, genome)
                # fitness_list.append(fitness_value)

            f_score = neat_instance.calculate_f_statistic(genome=neat_instance.best_all_time_genome,
                                                          x_test_data=neat_instance.x_test,
                                                          y_test_data=neat_instance.y_test)
            accuracy = neat_instance.calculate_accuracy(genome=neat_instance.best_all_time_genome,
                                                        x_test_data=neat_instance.x_test,
                                                        y_test_data=neat_instance.y_test)
            best_accuracy_list.append(accuracy)
            best_f1_list.append(f_score)
            best_fitness_list.append(neat_instance.best_all_time_genome.fitness)
            infile.close()
        except:
            pass

    print('AVG ACCURACY', np.mean(accuracy_list))
    print('AVG F1', np.mean(f1_score_list))
    # print('AVG FITNESS', np.mean(fitness_list))

    print('BEST ACCURACY', np.mean(best_accuracy_list))
    print('BEST F1', np.mean(best_f1_list))
    # print('BEST FITNESS', np.mean(best_fitness_list))


def plot_shm_multi_data(experiments_path):
    neat_instance_list = []
    generation_information_list = []
    x_test_data = None
    y_test_data = None
    predictions_list = []
    for directory in os.listdir(experiments_path):
        try:
            infile = open('{}/{}/NEAT_instance'.format(experiments_path, directory), 'rb')
            neat_instance = pickle.load(infile)
            neat_instance_list.append(neat_instance)
            connection_amount_list = []
            for genome in neat_instance.population.values():
                connection_amount_list.append(len(genome.connections))
            for index in range(len(connection_amount_list)):
                avg = np.mean(connection_amount_list)
                connection_amount_list[index] -= avg

            plt.bar(list(range(1, 17)), connection_amount_list)
            plt.show()
            if x_test_data is None:
                x_test_data = neat_instance.x_test
                y_test_data = neat_instance.y_test
            prediction = get_genome_predictions(genome=neat_instance.best_all_time_genome, x_data=x_test_data,
                                                round_values=False)
            predictions_list.append(prediction)
            infile.close()

            infile = open('{}/{}/generation_tracker'.format(experiments_path, directory), 'rb')
            generation_tracker_instance = pickle.load(infile)
            generation_information_list.append(generation_tracker_instance.generation_information)
            infile.close()
        except:
            pass
    generations_to_go_through = list(range(1, min(list(map(lambda x: max(x), generation_information_list))) + 1))

    gen_accuracy_list = []
    gen_fitness_list = []
    gen_population_fitness_list = []
    gen_species_amount_list = []
    # TODO: If you need to get accracy correct then manuall go through best genomes in history
    for index in generations_to_go_through:
        avg_tracker_accuracy = []
        avg_tracker_fitness = []
        avg_tracker_population_fitness = []
        avg_species_amount = []
        for run in generation_information_list:
            avg_tracker_accuracy.append(run[index]['best_all_time_genome_accuracy'])
            avg_tracker_fitness.append(run[index]['best_all_time_genome_fitness'])
            avg_tracker_population_fitness.append(run[index]['average_population_fitness'])
            avg_species_amount.append(run[index]['num_species'])

        gen_accuracy_list.append(np.mean(avg_tracker_accuracy))
        gen_fitness_list.append(np.mean(avg_tracker_fitness))
        gen_population_fitness_list.append(np.mean(avg_tracker_population_fitness))
        gen_species_amount_list.append(np.mean(avg_species_amount))

    plt.figure()
    plt.plot(generations_to_go_through, gen_species_amount_list, label='Number of Species')
    plt.ylabel('Number of Species')
    plt.xlabel('Generation')
    plt.show()

    # # Have to reset first one because its so large
    gen_fitness_list[0] = gen_fitness_list[1]

    plt.figure()
    label1 = plt.plot(generations_to_go_through, gen_accuracy_list, label='Best Genome Accuracy')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    # axes2.legend(loc='upper right')
    plt.xlabel('Generation')

    axes2 = plt.twinx()
    label2 = axes2.plot(generations_to_go_through, gen_fitness_list, color='darkorange', label='Best Genome Fitness')
    axes2.set_ylabel('Fitness')
    lns = label1 + label2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc=0)

    plt.show()

    gen_population_fitness_list[0] = gen_population_fitness_list[1]
    plt.figure()
    label1 = plt.plot(generations_to_go_through, gen_population_fitness_list, label='Average Population Fitness')
    plt.ylabel('Fitness')
    plt.legend(loc='lower right')
    # axes2.legend(loc='upper right')
    plt.xlabel('Generation')

    label2 = plt.plot(generations_to_go_through, gen_fitness_list, label='Best Genome Fitness')
    lns = label1 + label2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc=0)

    plt.show()

    avg_predictions_array = np.empty((len(predictions_list[0]), len(predictions_list[0][0])))
    for i in range(len(predictions_list[0])):
        avg_tracker = np.empty((len(predictions_list), len(predictions_list[0][0])))
        counter = 0
        for run in predictions_list:
            predictions = run[i, :]
            avg_tracker[counter, :] = predictions
            counter += 1
        avg_predictions_array[i, :] = np.sum(avg_tracker, axis=0)

    before_rounding = avg_predictions_array

    y_predicted = []
    y_actual = []

    for row in range(y_test_data.shape[0]):
        predicted = np.argmax(before_rounding[row, :])
        y_predicted.append(predicted)
        actual = np.argmax(y_test_data[row, :])
        y_actual.append(actual)

    wrong_counter = 0
    for i in range(len(y_actual)):
        if y_actual[i] != y_predicted[i]:
            wrong_counter += 1
        # Make states 1-17 instead of 0-16
        y_actual[i] += 1
        y_predicted[i] += 1
    print('wrong: ', wrong_counter)
    print('wrong percentage', wrong_counter / len(y_actual))
    data = {'y_Predicted': y_predicted,
            'y_Actual': y_actual,
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    sn.heatmap(confusion_matrix, annot=True, cmap="YlGnBu")
    plt.show()


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

    plot_data = False
    show_decision_boundary = False
    visualise_generation = False
    visualise_population_complexity = False
    get_table_values = False
    plot_confusion_matrix = False
    plot_figure_shm_data = False
    plot_figure_model_complexity_during_evolution = False
    plot_figure_shm_multi = True
    experiment_path = 'algorithm_runs\\xor_small_noise'
    # experiment_path = 'algorithm_runs\\shm_two_class'

    # PLOT DATA
    if plot_data:
        # TODO: Add legends
        colors = create_label_colours(labels=y_data)
        fig, ax = plt.subplots()
        for x1, x2, color in zip(feature_1_xor, feature_2_xor, colors):
            ax.scatter(x1, x2, label=color, )
        plt.title('XOR Data')
        plt.xlabel('X1')
        plt.ylabel('X2')
        ax.legend()
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
        plot_decision_boundary(experiments_path=experiment_path, data_being_used='xor_data')

    if visualise_generation:
        visualise_generation_tracker(experiments_path=experiment_path)
    if visualise_population_complexity:
        plot_population_complexity(experiments_path=experiment_path)

    if plot_confusion_matrix:
        create_confusion_matrix(x_data=x_data, y_data=y_data, experiments_path=experiment_path)

    if plot_figure_model_complexity_during_evolution:
        plot_model_complexity_during_evolution(experiments_path=experiment_path)

    if get_table_values:
        get_avg_table_values(experiments_path=experiment_path)

    if plot_figure_shm_data:
        plot_shm_data(rotation_angle=30, elevation=-160, experiments_path='algorithm_runs/shm_two_class')

    if plot_figure_shm_multi:
        plot_shm_multi_data(experiments_path='algorithm_runs_multi/shm_multi_class')


if __name__ == "__main__":
    main()
