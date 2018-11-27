from genome_neural_network import GenomeNeuralNetwork
from reproduce import Reproduce
from species import SpeciesSet

# Exception used to check if there are no more species
from stagnation import Stagnation


class CompleteExtinctionException(Exception):
    pass


class NEAT:

    def __init__(self, x_training_data, y_training_data, config, fitness_threshold):
        # Where all the parameters are saved
        self.config = config
        # Takes care of reproduction of populations
        self.reproduction = Reproduce(stagnation=Stagnation, config=config)
        # Track the best genome across generations
        self.best_all_time_genome = None
        # If the fitness threshold is met it will stop the algorithm (if set)
        self.fitness_threshold = fitness_threshold
        # A class containing the different species within the population
        self.species_set = SpeciesSet(config=config)
        self.x_train = x_training_data
        self.y_train = y_training_data

        # Initialise the starting population
        self.population = self.reproduction.create_new_population(population_size=self.config.population_size,
                                                                  num_features=x_training_data.shape[1])

        # Speciate the initial population
        self.species_set.speciate(population=self.population, compatibility_threshold=3, generation=0)

    def evaluate_population(self, use_backprop):
        """
        Calculates the fitness value for each individual genome in the population
        :type use_backprop: True or false on whether you're calculating the fitness using backprop or not
        :return: The best genome of the population
        """

        # Should return the best genome
        current_best_genome = None
        costs_for_current_gen = {}
        for genome in self.population.values():
            genome_nn = GenomeNeuralNetwork(genome=genome, x_train=self.x_train, y_train=self.y_train,
                                            create_weights_bias_from_genome=True, activation_type='sigmoid',
                                            learning_rate=0.0001, num_epochs=1000, batch_size=64)

            # Optimise the neural_network_first
            if use_backprop:
                genome_nn.optimise()

            # We use genome_nn.x_train instead of self.x_train because the genome_nn might have deleted a row if there
            # is no connection to one of the sources
            cost = genome_nn.run_one_pass(input_data=genome_nn.x_train, labels=self.y_train, return_cost_only=True)

            # This shouldn't ever happen because due to the floats being different values and the weights being
            # different for every genome
            if cost in costs_for_current_gen:
                costs_for_current_gen[cost] += 1
            else:
                costs_for_current_gen[cost] = 1

            # The fitness is the negative of the cost. Because less cost = greater fitness
            genome.fitness = -cost

            if current_best_genome is None or genome.fitness > current_best_genome.fitness:
                current_best_genome = genome

        return current_best_genome

    def run(self, max_num_generations, use_backprop):
        """
        Run the algorithm
        """

        current_gen = 0
        while current_gen < max_num_generations:
            # Every generation increment
            current_gen += 1

            # Evaluate the current generation and get the best genome in the current generation
            best_current_genome = self.evaluate_population(use_backprop=use_backprop)

            # Keep track of the best genome across generations
            if self.best_all_time_genome is None or best_current_genome.fitness > self.best_all_time_genome.fitness:
                self.best_all_time_genome = best_current_genome

            # If the fitness threshold is met, stop the algorithm
            if self.best_all_time_genome.fitness > self.fitness_threshold:
                break

            # Reproduce and get the next generation
            self.population = self.reproduction.reproduce(species_set=self.species_set,
                                                          population_size=self.config.population_size,
                                                          generation=current_gen)

            # Allow for some leaway in population size (+- 5)
            range_value = 5
            range_of_population_sizes = set(range(self.config.population_size - range_value,
                                                      self.config.population_size + range_value + 1))
            if len(self.population) not in range_of_population_sizes:
                raise Exception('There is an incorrect number of genomes in the population')

            # Check to ensure no genes share the same connection gene addresses
            self.ensure_no_duplicate_genes()

            # Check if there are any species, if not raise an exception. TODO: Let user reset population if extinction
            if not self.species_set.species:
                raise CompleteExtinctionException()

            # Speciate the current generation
            self.species_set.speciate(population=self.population, generation=current_gen,
                                      compatibility_threshold=self.config.compatibility_threshold)

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
