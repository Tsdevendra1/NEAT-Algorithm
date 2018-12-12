import matplotlib.pyplot as plt


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

    def update_generation_information(self, generation):
        print('\n')

        information = {'mean_num_nodes_overall': round(self.mean_number_nodes_overall, 2),
                       'mean_number_nodes_enabled': round(self.mean_number_nodes_enabled, 2),
                       'mean_number_connections_overall': round(self.mean_number_connections_overall, 2),
                       'mean_number_connections_enabled': round(self.mean_number_connections_enabled, 2),
                       'average_population_fitness': round(self.average_population_fitness, 3),
                       'mean_compatibility_distance': round(self.mean_compatibility_distance, 2),
                       'species_execute_time': round(self.species_execute_time, 2),
                       'population_size': round(self.population_size, 2),
                       'best_all_time_genome_fitness': round(self.best_all_time_genome_fitness, 3),
                       'reproduce_excute_time': round(self.reproduce_execute_time, 2),
                       'evaluate_execute_time': round(self.evaluate_execute_time, 2),
                       'num_species': round(self.num_species, 2),
                       'std_dev_compatibility_distance': round(self.std_dev_compatibility_distance, 2)}

        self.generation_information[generation] = information

    def print_generation_information(self, generation_interval_for_graph):
        current_gen = max(self.generation_information)
        print('\n' * 2)
        print('For the current generation number: {}'.format(current_gen))

        information_dict = {'mean_num_nodes_overall': round(self.mean_number_nodes_overall, 2),
                            'mean_number_nodes_enabled': round(self.mean_number_nodes_enabled, 2),
                            'mean_number_connections_overall': round(self.mean_number_connections_overall, 2),
                            'mean_number_connections_enabled': round(self.mean_number_connections_enabled, 2),
                            'average_population_fitness': round(self.average_population_fitness, 3),
                            'mean_compatibility_distance': round(self.mean_compatibility_distance, 2),
                            'best_all_time_genome_fitness': round(self.best_all_time_genome_fitness, 3),
                            'species_execute_time': round(self.species_execute_time, 2),
                            'reproduce_excute_time': round(self.reproduce_execute_time, 2),
                            'population_size': round(self.population_size, 2),
                            'evaluate_execute_time': round(self.evaluate_execute_time, 2),
                            'num_species': round(self.num_species, 2),
                            'std_dev_compatibility_distance': round(self.std_dev_compatibility_distance, 2)}

        for information_type, information in information_dict.items():
            print(information_type, ':', ' {}'.format(information))
            if current_gen % generation_interval_for_graph == 0:
                generations_to_go_through = list(range(1, current_gen+1))
                y_data = []
                for generation in generations_to_go_through:
                    y_data.append(self.generation_information[generation][information_type])

                plt.plot(generations_to_go_through, y_data)
                plt.title(information_type)
                plt.show()
