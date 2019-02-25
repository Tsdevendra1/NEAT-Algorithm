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

    def update_generation_information(self, generation):

        # information = {'mean_num_nodes_overall': round(self.mean_number_nodes_overall, 2),
        #                'mean_number_nodes_enabled': round(self.mean_number_nodes_enabled, 2),
        #                'mean_number_connections_overall': round(self.mean_number_connections_overall, 2),
        #                'mean_number_connections_enabled': round(self.mean_number_connections_enabled, 2),
        #                'average_population_fitness': round(self.average_population_fitness, 3),
        #                'mean_compatibility_distance': round(self.mean_compatibility_distance, 2),
        #                'species_execute_time': round(self.species_execute_time, 2),
        #                'population_size': round(self.population_size, 2),
        #                'best_all_time_genome_fitness': round(self.best_all_time_genome_fitness, 3),
        #                'reproduce_excute_time': round(self.reproduce_execute_time, 2),
        #                'evaluate_execute_time': round(self.evaluate_execute_time, 2),
        #                'num_species': round(self.num_species, 2),
        #                'num_generation_add_node': self.num_generation_add_node,
        #                'num_generation_delete_node': self.num_generation_delete_node,
        #                'num_generation_add_connection': self.num_generation_add_connection,
        #                'num_generation_delete_connection': self.num_generation_delete_connection,
        #                'std_dev_compatibility_distance': round(self.std_dev_compatibility_distance, 2)}

        information = {}
        for info_type, info_value in self.__dict__.items():
            if isinstance(info_value, float) or isinstance(info_value, np.float64):
                information[info_type] = round(info_value, 2)
            else:
                information[info_type] = info_value

        self.generation_information[generation] = information

    def print_generation_information(self, generation_interval_for_graph):
        current_gen = max(self.generation_information.keys())
        print('**************************** Generation {} *******************************'.format(current_gen))

        important_information = [
            ('Number of Species', self.generation_information[current_gen]['num_species']),
            ('Added Node Mutations', self.generation_information[current_gen]['num_generation_add_node']),
            ('Delete Node Mutations', self.generation_information[current_gen]['num_generation_delete_node']),
            ('Add Connection Mutations', self.generation_information[current_gen]['num_generation_add_connection']),
            ('Delete Connection Mutations', self.generation_information[current_gen]['num_generation_delete_connection']),
            ('Average Fitness', self.generation_information[current_gen]['average_population_fitness']),
            ('Best Genome Fitness', self.generation_information[current_gen]['best_all_time_genome_fitness']),
            ('Average Number of Connections Per Genome', self.generation_information[current_gen]['mean_number_connections_enabled']),
            ('Average Number of Nodes Per Genome', self.generation_information[current_gen]['mean_number_nodes_enabled']),
            ('Average Compatibility Distance', self.generation_information[current_gen]['mean_compatibility_distance']),
        ]

        important_information = collections.OrderedDict(important_information)

        for info_type, info_value in important_information.items():
            print('{}:{}'.format(info_type, info_value))
        print('\n')

        # Plot information to graph
        for information_type, information in self.generation_information[current_gen].items():
            # Don't need to print the dictionary
            if information_type != 'generation_information':
                # print(information_type, ':', ' {}'.format(information))
                if current_gen % generation_interval_for_graph == 0:
                    generations_to_go_through = list(range(1, current_gen + 1))
                    y_data = []
                    for generation in generations_to_go_through:
                        y_data.append(self.generation_information[generation][information_type])

                    plt.plot(generations_to_go_through, y_data)
                    plt.title(information_type)
                    plt.show()
