from generation_statistics import GenerationStatistics
import time
import numpy as np
from genome_neural_network import GenomeNeuralNetwork
from gene import NodeGene, ConnectionGene
from reproduce import Reproduce
from genome import Genome
from species import SpeciesSet
import sklearn.metrics
import pickle

# Exception used to check if there are no more species
from stagnation import Stagnation


class CompleteExtinctionException(Exception):
    pass


class NEAT:

    def __init__(self, x_training_data, y_training_data, x_test_data, y_test_data, config, fitness_threshold,
                 f1_score_threshold, algorithm_running=None):
        # Where all the parameters are saved
        self.config = config
        # Takes care of reproduction of populations
        self.reproduction = Reproduce(stagnation=Stagnation, config=config)
        self.generation_tracker = GenerationStatistics()
        # Track the best genome across generations
        self.best_all_time_genome = None
        # If the fitness threshold is met it will stop the algorithm (if set)
        self.fitness_threshold = fitness_threshold
        self.f1_score_threshold = f1_score_threshold
        # A class containing the different species within the population
        self.species_set = SpeciesSet(config=config, generation_tracker=self.generation_tracker)
        self.x_train = x_training_data
        self.y_train = y_training_data
        self.x_test = x_test_data
        self.y_test = y_test_data

        if algorithm_running:
            # Defines which of the algorithms is being currently tested (e.g. xor with 5000 examples of xor with 200
            # examples and noise)
            self.algorithm_running = algorithm_running

        # Initialise the starting population
        self.population = self.reproduction.create_new_population(population_size=self.config.population_size,
                                                                  num_features=x_training_data.shape[1])

        # Speciate the initial population
        self.species_set.speciate(population=self.population, compatibility_threshold=3, generation=0)

    @staticmethod
    def create_genome_nn(genome, x_data, y_data, algorithm_running=None):
        # TODO: I encountered a bug where I trained a genome on a relu activation function, but when I recreated using this function I had problems because I forgot that everything defined inside here uses sigmoid. Should improve implementation of this
        # TODO: The x_data, y_data isn't always used, particularly if we only create the network to get a prediction. This implementation should be improved for clarity
        if algorithm_running == 'xor_full':
            learning_rate = 0.1
            num_epochs = 500
            batch_size = 64
            activation_type = 'sigmoid'
        elif algorithm_running == 'xor_small_noise':
            learning_rate = 0.1
            num_epochs = 5000
            batch_size = 10
            activation_type = 'sigmoid'
        elif algorithm_running == 'circle_data':
            learning_rate = 0.1
            num_epochs = 5000
            batch_size = 50
            activation_type = 'sigmoid'
        elif algorithm_running == 'shm_two_class':
            learning_rate = 0.1
            num_epochs = 5000
            batch_size = 50
            activation_type = 'sigmoid'
        # TODO: Choose more suitable default
        else:
            learning_rate = 0.1
            num_epochs = 500
            batch_size = 64
            activation_type = 'sigmoid'

        return GenomeNeuralNetwork(genome=genome, x_train=x_data, y_train=y_data,
                                   create_weights_bias_from_genome=True, activation_type=activation_type,
                                   learning_rate=learning_rate, num_epochs=num_epochs, batch_size=batch_size)

    def evaluate_population(self, use_backprop, generation):
        """
        Calculates the fitness value for each individual genome in the population
        :type use_backprop: True or false on whether you're calculating the fitness using backprop or not
        :param generation: Which generation number it currently is
        :return: The best genome of the population
        """

        # Should return the best genome
        current_best_genome = None
        current_worst_genome = None

        for genome in self.population.values():

            genome_nn = self.create_genome_nn(genome=genome, x_data=self.x_train, y_data=self.y_train,
                                              algorithm_running=self.algorithm_running)

            # Optimise the neural_network_first. However, the generation should allow for one pass so that we are not
            #  just optimising all the same topologies
            genome_fitness_before = genome.fitness
            if use_backprop and generation > 1:
                print('\n')
                print('OPTIMISING GENOME')
                genome_nn.optimise(print_epoch=False)

            # We use genome_nn.x_train instead of self.x_train because the genome_nn might have deleted a row if there
            # is no connection to one of the sources
            cost = genome_nn.run_one_pass(input_data=genome_nn.x_train, labels=self.y_train, return_cost_only=True)

            # The fitness is the negative of the cost. Because less cost = greater fitness
            genome.fitness = -cost

            # Only print genome fitness after is back prop is used since back prop takes a long time so this can be a
            #  way of tracking progress in the meantime
            if use_backprop and generation > 1:
                # NOTE: Genome fitness can be none due to crossover because fitness value not carried over
                print('Genome Fitness Before: {}'.format(genome_fitness_before))
                print('Genome Fitness After: {}'.format(genome.fitness))

            if current_best_genome is None or genome.fitness > current_best_genome.fitness:
                current_best_genome = genome
            if current_worst_genome is None or genome.fitness < current_worst_genome.fitness:
                current_worst_genome = genome

        return current_best_genome, current_worst_genome

    def update_population_toplogy_info(self):
        num_nodes_overall = []
        num_nodes_enabled = []
        num_connections_overall = []
        num_connections_enabled = []
        all_fitnesses = []
        for genome in self.population.values():
            num_nodes_overall.append(len(genome.nodes))
            num_nodes_enabled.append(len(genome.get_active_nodes()))
            num_connections_overall.append(len(genome.connections))
            num_connections_enabled.append(genome.check_connection_enabled_amount())
            if genome.fitness:
                all_fitnesses.append(genome.fitness)

        self.generation_tracker.mean_number_connections_enabled = np.mean(num_connections_enabled)
        self.generation_tracker.mean_number_connections_overall = np.mean(num_connections_overall)

        self.generation_tracker.mean_number_nodes_enabled = np.mean(num_nodes_enabled)
        self.generation_tracker.mean_number_nodes_overall = np.mean(num_nodes_overall)

        self.generation_tracker.average_population_fitness = np.mean(all_fitnesses)

    def add_successful_genome_for_test(self, current_gen, use_this_genome):
        """
        This function adds a pre programmed genome which is known to converge for the XOR dataset.
        :param current_gen:
        :param use_this_genome: Whether this genome should be added to the population or not
        :return:
        """
        # Wait for current_gen > 1 because if using backprop the first gen skips using backprop.
        if current_gen > 1 and use_this_genome:
            node_list = [
                NodeGene(node_id=0, node_type='source'),
                NodeGene(node_id=1, node_type='source'),
                NodeGene(node_id=2, node_type='output', bias=0.5),
                NodeGene(node_id=3, node_type='hidden', bias=1),
                NodeGene(node_id=4, node_type='hidden', bias=1),
                NodeGene(node_id=5, node_type='hidden', bias=1),
                NodeGene(node_id=6, node_type='hidden', bias=1),
            ]

            connection_list = [ConnectionGene(input_node=0, output_node=3, innovation_number=1, enabled=True,
                                              weight=np.random.randn()),
                               ConnectionGene(input_node=1, output_node=3, innovation_number=2, enabled=True,
                                              weight=np.random.randn()),
                               ConnectionGene(input_node=0, output_node=4, innovation_number=3, enabled=True,
                                              weight=np.random.randn()),
                               ConnectionGene(input_node=1, output_node=4, innovation_number=4, enabled=True,
                                              weight=np.random.randn()),
                               ConnectionGene(input_node=3, output_node=5, innovation_number=5, enabled=True,
                                              weight=np.random.randn()),
                               ConnectionGene(input_node=4, output_node=5, innovation_number=6, enabled=True,
                                              weight=np.random.randn()),
                               ConnectionGene(input_node=3, output_node=6, innovation_number=7, enabled=True,
                                              weight=np.random.randn()),
                               ConnectionGene(input_node=4, output_node=6, innovation_number=8, enabled=True,
                                              weight=np.random.randn()),
                               ConnectionGene(input_node=5, output_node=2, innovation_number=9, enabled=True,
                                              weight=np.random.rand()),
                               ConnectionGene(input_node=6, output_node=2, innovation_number=10, enabled=True,
                                              weight=np.random.randn())
                               ]

            test_genome = Genome(connections=connection_list, nodes=node_list, key=1)
            test_genome.fitness = -99999999999
            self.population[32131231] = test_genome

    @staticmethod
    def calculate_f_statistic(genome, x_test_data, y_test_data):
        genome_nn = NEAT.create_genome_nn(genome=genome, x_data=x_test_data, y_data=y_test_data)
        prediction = genome_nn.run_one_pass(input_data=x_test_data, return_prediction_only=True).round()
        return sklearn.metrics.f1_score(y_test_data, prediction)

    def run(self, max_num_generations, use_backprop, print_generation_information, show_population_weight_distribution):
        """
        Run the algorithm
        """

        current_gen = 0
        while current_gen < max_num_generations:
            # Every generation increment
            current_gen += 1

            self.add_successful_genome_for_test(current_gen=current_gen, use_this_genome=False)

            self.generation_tracker.population_size = len(self.population)

            start_evaluate_time = time.time()
            # Evaluate the current generation and get the best genome in the current generation
            best_current_genome, worst_current_genome = self.evaluate_population(use_backprop=use_backprop,
                                                                                 generation=current_gen)
            print('WORST CURRENT GENOME FITNESS: {}'.format(worst_current_genome.fitness))
            end_evaluate_time = time.time()
            self.update_population_toplogy_info()
            self.generation_tracker.evaluate_execute_time = end_evaluate_time - start_evaluate_time

            # Keep track of the best genome across generations
            if self.best_all_time_genome is None or best_current_genome.fitness > self.best_all_time_genome.fitness:
                self.best_all_time_genome = best_current_genome

            self.generation_tracker.best_all_time_genome_fitness = self.best_all_time_genome.fitness

            start_reproduce_time = time.time()

            # Reset attributes for the current generation
            self.generation_tracker.reset_tracker_attributes()

            # Reproduce and get the next generation
            self.population = self.reproduction.reproduce(species_set=self.species_set,
                                                          population_size=self.config.population_size,
                                                          generation=current_gen,
                                                          generation_tracker=self.generation_tracker,
                                                          # current_gen should be greater than one ot use
                                                          # backprop_mutation because we let the first generation
                                                          # mutate just as if it was the normal genetic algorithm,
                                                          # so that we're not optimising all of the same structure
                                                          backprop_mutation=(use_backprop and current_gen > 1))
            end_reproduce_time = time.time()
            self.generation_tracker.reproduce_execute_time = end_reproduce_time - start_reproduce_time

            # TODO: Uncomment this if population sizes causes an issue
            # # Allow for some leaway in population size (+- 5)
            # range_value = 5
            # range_of_population_sizes = set(range(self.config.population_size - range_value,
            #                                       self.config.population_size + range_value + 1))
            # if len(self.population) not in range_of_population_sizes:
            #     raise Exception('There is an incorrect number of genomes in the population')

            # Check to ensure no genes share the same connection gene addresses. (This problem has been fixed but is
            # here just incase now).
            self.ensure_no_duplicate_genes()

            # Check if there are any species, if not raise an exception. TODO: Let user reset population if extinction
            if not self.species_set.species:
                raise CompleteExtinctionException()

            start_specify_time = time.time()
            # Speciate the current generation
            self.species_set.speciate(population=self.population, generation=current_gen,
                                      compatibility_threshold=self.config.compatibility_threshold,
                                      generation_tracker=self.generation_tracker)
            end_specify_time = time.time()
            self.generation_tracker.species_execute_time = end_specify_time - start_specify_time

            f1_score_of_best_all_time_genome = self.calculate_f_statistic(
                self.best_all_time_genome, self.x_test, self.y_test)

            self.generation_tracker.best_all_time_genome_f1_score = f1_score_of_best_all_time_genome

            self.generation_tracker.update_generation_information(generation=current_gen)

            if print_generation_information:
                self.generation_tracker.print_generation_information(generation_interval_for_graph=1, plot_graphs=False)

            # If the fitness threshold is met, stop the algorithm
            if self.best_all_time_genome.fitness > self.fitness_threshold or f1_score_of_best_all_time_genome > self.f1_score_threshold:
                # Save best genome in pickle
                outfile = open('pickles/best_genome_pickle_{}_{}'.format(self.algorithm_running,
                                                                         np.random.random_integers(low=0, high=10)),
                               'wb')
                pickle.dump(self.best_all_time_genome, outfile)
                outfile.close()
                self.generation_tracker.plot_graphs(current_gen=current_gen)
                break

            # Gives distribution of the weights in the population connections
            if show_population_weight_distribution:
                self.reproduction.show_population_weight_distribution(population=self.population)

        print('f1 score for best genome after optimising is: {}'.format(f1_score_of_best_all_time_genome))

        return self.best_all_time_genome

    def ensure_no_duplicate_genes(self):
        connection_gene_dict = {}
        for genome in self.population.values():
            for connection in genome.connections.values():
                if connection not in connection_gene_dict:
                    connection_gene_dict[connection] = 1
                else:
                    connection_gene_dict[connection] += 1

        for connection_gene, amount in connection_gene_dict.items():
            if amount > 1:
                raise Exception('You have duplicated a connection gene')
