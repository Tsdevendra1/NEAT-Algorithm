from genome_neural_network import GenomeNeuralNetwork
from reproduce import Reproduce
from species import SpeciesSet


# Exception used to check if there are no more species
class CompleteExtinctionException(Exception):
    pass


class NEAT:

    def __init__(self, x_training_data, y_training_data, fitness_threshold=None):
        self.population = None
        self.reproduction = Reproduce()
        # Track the best genome across generations
        self.best_all_time_genome = None
        # If the fitness threshold is met it will stop the algorithm (if set)
        self.fitness_threshold = fitness_threshold
        self.species = SpeciesSet()
        self.x_train = x_training_data
        self.y_train = y_training_data

    def evaluate_population(self, use_backprop):
        """
        Calculates the fitness value for each individual genome in the population
        :return:
        """

        # Should return the best genome
        current_best_genome = None
        for genome in self.population:
            genome_nn = GenomeNeuralNetwork(genome=genome, x_train=self.x_train, y_train=self.y_train,
                                            create_weights_bias_from_genome=True, activation_type='sigmoid',
                                            learning_rate=0.0001, num_epochs=1000, batch_size=64)
            # Optimise the neural_network_first
            if use_backprop:
                genome_nn.optimise()

            cost = genome_nn.run_one_pass(input_data=self.x_train, labels=self.y_train)
            # The fitness is the negative of the cost. Because less cost = greater fitness
            genome.fitness = -cost

            if current_best_genome is None or genome.fitness > current_best_genome.fitness:
                current_best_genome = genome

        return current_best_genome

    def run(self, max_num_generations):
        """
        Run the algorithm
        """

        current_gen = 0
        while current_gen < max_num_generations:
            # Every generation increment
            current_gen += 1

            # Evaluate the current generation and get the best genome in the current generation
            best_current_genome = self.evaluate_population(use_backprop=False)

            # Keep track of the best genome across generations
            if self.best_all_time_genome is None or best_current_genome.fitness > self.best_all_time_genome.fitness:
                self.best_all_time_genome = best_current_genome

            # If the fitness threshold is met, stop the algorithm
            if self.best_all_time_genome.fitness > self.fitness_threshold:
                break

            # Reproduce and get the next generation
            self.population = self.reproduction.reproduce(species_instance=self.species,
                                                          population_size=len(self.population))

            # Check if there are any species, if not raise an exception. TODO: Let user reset population if extinction
            if not self.species.species:
                raise CompleteExtinctionException()

            # Speciate the current generation
            self.species.speciate(population=self.population, generation=current_gen)

        return self.best_all_time_genome
