import matplotlib.pyplot as plt
import numpy as np
import collections


class GenerationStatistics:

    def __init__(self):
        self.generation_information = {}
        self.mean_compatibility_distance = None
        self.std_dev_compatibility_distance = None
        self.best_all_time_genome_fitness = None
        self.average_population_fitness = None
        self.num_species = None
        self.mean_number_connections_overall = None
        self.mean_number_connections_enabled = None
        self.population_size = None
        self.mean_number_nodes_overall = None
        self.mean_number_nodes_enabled = None
        self.species_execute_time = None
        self.reproduce_execute_time = None
        self.evaluate_execute_time = None
        self.num_generation_add_node = None
        self.num_generation_delete_node = None
        self.num_generation_add_connection = None
        self.num_generation_delete_connection = None
        self.num_generation_weight_mutations = None
        self.perturbation_values_max = None
        self.perturbation_values_min = None
        self.perturbation_values_list = None
        self.num_disjoint_list = None
        self.num_excess_list = None
        self.weight_diff_list = None
        self.avg_num_disjoint = None
        self.avg_num_excess = None
        self.avg_weight_diff = None
        self.best_all_time_genome_f1_score = None
        self.best_all_time_genome_accuracy = None

    def update_generation_information(self, generation):

        # Update min and max values of perturbation to weights
        self.perturbation_values_max = max(self.perturbation_values_list)
        self.perturbation_values_min = min(self.perturbation_values_list)
        self.avg_num_disjoint = np.mean(self.num_disjoint_list)
        self.avg_num_excess = np.mean(self.num_excess_list)
        self.avg_weight_diff = np.mean(self.weight_diff_list)

        information = {}
        for info_type, info_value in self.__dict__.items():
            if isinstance(info_value, float) or isinstance(info_value, np.float64):
                information[info_type] = round(info_value, 2)
            else:
                information[info_type] = info_value

        self.generation_information[generation] = information

    def reset_tracker_attributes(self):
        """
        Reset the number of mutations which have occured for the current generation.
        :return:
        """
        self.num_generation_add_connection = 0
        self.num_generation_add_node = 0
        self.num_generation_delete_connection = 0
        self.num_generation_delete_node = 0
        self.num_generation_weight_mutations = 0
        self.perturbation_values_list = []
        self.num_excess_list = []
        self.num_disjoint_list = []
        self.weight_diff_list = []

    def plot_graphs(self, current_gen, save_plots=False, file_path=None):

        if (save_plots and not file_path) or (file_path and not save_plots):
            raise Exception('Save_plots and file_paths must be specified at the same time')

        important_information_keys = {
            'num_species', 'num_generation_add_node', 'num_generation_delete_node', 'num_generation_add_connection',
            'num_generation_delete_connection', 'num_generation_weight_mutations', 'average_population_fitness',
            'best_all_time_genome_fitness', 'mean_number_connections_enabled', 'mean_number_nodes_enabled',
            'mean_compatibility_distance', 'avg_num_disjoint', 'avg_num_excess', 'avg_weight_diff',
            'mean_number_connections_overall', 'best_all_time_genome_f1_score', 'best_all_time_genome_accuracy'
        }

        # Plot information to graph every certain amount of generations
        # for information_type, information in self.generation_information[current_gen].items():
        for information_type in important_information_keys:
            # Don't need to print the dictionary
            if information_type != 'generation_information':
                # print(information_type, ':', ' {}'.format(information))
                # if current_gen % generation_interval_for_graph == 0 and current_gen != 1:
                generations_to_go_through = list(range(1, current_gen + 1))
                y_data = []
                for generation in generations_to_go_through:
                    y_data.append(self.generation_information[generation][information_type])

                plt.plot(generations_to_go_through, y_data)
                plt.title(information_type)
                if save_plots:
                    plt.savefig('{}/{}_generation_{}.png'.format(file_path, information_type, current_gen))
                plt.show()

    def print_generation_information(self, generation_interval_for_graph, plot_graphs_every_gen):
        current_gen = max(self.generation_information.keys())
        print('**************************** Generation {} *******************************'.format(current_gen))

        important_information = [
            ('Number of Species', self.generation_information[current_gen]['num_species']),
            ('Added Node Mutations', self.generation_information[current_gen]['num_generation_add_node']),
            ('Delete Node Mutations', self.generation_information[current_gen]['num_generation_delete_node']),
            ('Add Connection Mutations', self.generation_information[current_gen]['num_generation_add_connection']),
            ('Delete Connection Mutations',
             self.generation_information[current_gen]['num_generation_delete_connection']),
            ('Weight Mutations', self.generation_information[current_gen]['num_generation_weight_mutations']),
            ('Average Fitness', self.generation_information[current_gen]['average_population_fitness']),
            ('Best All Time Genome Fitness', self.generation_information[current_gen]['best_all_time_genome_fitness']),
            (
                'Best All Time Genome f1 score',
                self.generation_information[current_gen]['best_all_time_genome_f1_score']),
            (
                'Best All Time Genome Accuracy Percent',
                self.generation_information[current_gen]['best_all_time_genome_accuracy']),

            ('Average Number of Connections Per Genome',
             self.generation_information[current_gen]['mean_number_connections_enabled']),
            ('Average Number of Nodes Per Genome',
             self.generation_information[current_gen]['mean_number_nodes_enabled']),
            ('Average Compatibility Distance', self.generation_information[current_gen]['mean_compatibility_distance']),
            ('Perturbation Max Value', self.generation_information[current_gen]['perturbation_values_max']),
            ('Perturbation Min Value', self.generation_information[current_gen]['perturbation_values_min']),
            ('Average Number of Disjoint Genes', self.generation_information[current_gen]['avg_num_disjoint']),
            ('Average Number of Excess Genes', self.generation_information[current_gen]['avg_num_excess']),
            ('Average Weight Difference', self.generation_information[current_gen]['avg_weight_diff']),
            ('Average Number of Connections',
             self.generation_information[current_gen]['mean_number_connections_overall']),
            # ('Average Number of Nodes', self.generation_information[current_gen]['avg_weight_diff']),
        ]

        # Make it an ordereddict to keep the order above.
        important_information = collections.OrderedDict(important_information)

        # Print the information
        for info_type, info_value in important_information.items():
            print('{}:{}'.format(info_type, info_value))
        print('\n')

        if current_gen % generation_interval_for_graph == 0 and current_gen != 1 and plot_graphs_every_gen:
            self.plot_graphs(current_gen=current_gen)
