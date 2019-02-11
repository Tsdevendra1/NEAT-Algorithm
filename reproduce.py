from gene import NodeGene, ConnectionGene
from genome import Genome
from species import SpeciesSet
import random
import numpy as np
import math
import copy


class Reproduce:
    """
    Will handle reproduction of the genomes
    """

    def __init__(self, stagnation, config):
        self.global_innovation_number = 0
        self.stagnation = stagnation
        self.ancestors = {}
        self.genome_indexer = 0
        self.config = config
        # Key: The tuple of the connection e.g. (1,3) value: the innovation number
        self.innovation_tracker = {}

    def create_new_population(self, population_size, num_features):
        population = {}

        node_list = []
        connection_list = []
        # Create the source nodes
        for node in range(num_features):
            node_list.append(NodeGene(node_id=node, node_type='source'))

        # Add the output node
        node_list.append(NodeGene(node_id=num_features, node_type='output', bias=0))

        # Save the innovations for the first generation.
        for source_node_id in range(num_features):
            # Increment for the new innovation
            self.global_innovation_number += 1
            # The output node will always have the node_id equal to the number of features
            self.innovation_tracker[(source_node_id, num_features)] = self.global_innovation_number

        # For each feature there will be a connection to the output
        for i in range(num_features):
            connection = (i, num_features)
            # The connection was already saved, so this should return true
            assert (connection in self.innovation_tracker)
            connection_list.append(ConnectionGene(input_node=i, output_node=num_features,
                                                  innovation_number=self.innovation_tracker[connection], enabled=True))

        # Create a population of size population_size
        for index in range(population_size):
            # Deep copies otherwise changing the connection weight change's it for every genome that has the same
            # reference to the class
            deep_copy_connections = copy.deepcopy(connection_list)
            deep_copy_nodes = copy.deepcopy(node_list)
            # Set all the connections to a random weight for each genome
            for connection in deep_copy_connections:
                connection.weight = np.random.randn()
            # Increment since the index value has been assigned
            self.genome_indexer += 1

            # Create the genome
            population[index] = Genome(connections=deep_copy_connections, nodes=deep_copy_nodes,
                                       key=self.genome_indexer)

        return population

    @staticmethod
    def compute_adjusted_species_sizes(adjusted_species_fitnesses, previous_species_sizes, population_size,
                                       min_species_size):
        """
        Compute the number of offspring per species, proportional to their fitnesses (See page 110 of NEAT paper)
        :param adjusted_species_fitnesses:
        :param previous_species_sizes:
        :param population_size:
        :param min_species_sizes:
        :return:
        """

        # Sum all the remaining adjusted species fitnesses
        adjusted_fitness_sum = sum(adjusted_species_fitnesses)

        adjusted_species_sizes = []

        for adjusted_fitness, previous_size in zip(adjusted_species_fitnesses, previous_species_sizes):
            if adjusted_fitness_sum is not None:
                # Calculate the adjusted species size for how much of the overall fitness they account for. If this
                # value is less than the min_species_size then we set it to that instead
                species_size = max(min_species_size, ((adjusted_fitness / adjusted_fitness_sum) * population_size))

            else:
                species_size = min_species_size

            # TODO: I'm not sure what this is mean't to do?
            # difference = (species_size - previous_size) * 0.5
            # rounded_difference = int(round(difference))
            #
            # adjusted_size = previous_size
            #
            # if abs(rounded_difference) > 0:
            #     adjusted_size += rounded_difference
            # elif difference > 0:
            #     adjusted_size += 1
            # elif difference < 0:
            #     adjusted_size -= 1

            adjusted_species_sizes.append(round(species_size))

        # There may not be the exact number of population size due to floating point precision. So instead we give this
        # extra to the best species
        if sum(adjusted_species_sizes) < population_size:
            max_fitness_index = adjusted_species_fitnesses.index(max(adjusted_species_fitnesses))
            adjusted_species_sizes[max_fitness_index] += 1

        # Allow for some leaway in population size (+- 5)
        range_value = 5
        range_of_population_sizes = set(range(population_size - range_value,
                                              population_size + range_value + 1))
        # TODO: Re-enabled this is problem occurs
        # if sum(adjusted_species_sizes) not in range_of_population_sizes:
        #     raise Exception('There is an incorrect number of genomes in the population')

        return adjusted_species_sizes

    def get_non_stagnant_species(self, species_set, generation):
        """
        Checks which species are stagnant ant returns the ones which aren't
        :param generation: Which generation number it is
        :param species_set: The species set instance which stores all the species
        :return: A list of non stagnant species
        """
        # Keeps track of all the fitnesses for the genomes in the population
        all_fitnesses = []
        # Keeps track of the species which aren't stagnant
        remaining_species = []

        # (Id, species instance, boolean)
        for species_id, species, is_stagnant in self.stagnation.update(species_set=species_set, generation=generation,
                                                                       config=self.config):
            # if is_stagnant:
            #     # TODO: What to do here??
            #     pass
            # else:
            #     # Save all the fitness in the species that isn't stagnant
            #     all_fitnesses += [member.fitness for member in species.members.values()]
            #     remaining_species.append(species)

            # Only save species if it is not stagnant
            if not is_stagnant:
                # Save all the fitness in the species that isn't stagnant
                all_fitnesses += [member.fitness for member in species.members.values()]
                remaining_species.append(species)

        # The case where there are no species left
        if not remaining_species:
            # TODO: Would this ever come here?
            raise Exception('There are no remaining species in the reproduce function')

        return all_fitnesses, remaining_species

    def get_adjusted_species_sizes(self, all_fitnesses, remaining_species, population_size):
        """
        Adjusts the size of the species for their fitness values
        :param all_fitnesses: A list of all fitness values for all genomes in the population
        :param remaining_species: A list of species which aren't stagnant
        :param population_size: The population size
        :return: A list of sizes for the new remaining species, adjusted for their respective fitness values
        """

        # Find min and max fitness across the entire population. We use this for explicit fitness sharing.
        min_genome_fitness = min(all_fitnesses)
        max_genome_fitness = max(all_fitnesses)

        fitness_range = max(1.0, max_genome_fitness - min_genome_fitness)

        # TODO: Not sure if this is the right method to do adjusted fitness
        for species in remaining_species:
            # The adjusted fitness is the mean of the species members fitnesses TODO: Is this correct?
            species.adjusted_fitness = np.mean([member.fitness for member in species.members.values()])

            # TODO: comment back in if correct
            # mean_species_fitness = np.mean([member.fitness for member in species.members.values()])
            # adjusted_fitness = (mean_species_fitness - min_genome_fitness) / fitness_range

        adjusted_species_fitnesses = [species.adjusted_fitness for species in remaining_species]

        # Get a list of the amount of members in each of the remaining species
        previous_species_sizes = [len(species.members) for species in remaining_species]

        adjusted_species_sizes = self.compute_adjusted_species_sizes(
            adjusted_species_fitnesses=adjusted_species_fitnesses, min_species_size=self.config.min_species_size,
            previous_species_sizes=previous_species_sizes, population_size=population_size)

        return adjusted_species_sizes

    def get_new_population(self, adjusted_species_sizes, remaining_species, species_set):
        """
        Creates the dictionary of the new genomes for the next generation population
        :param adjusted_species_sizes:
        :param remaining_species:
        :param species_set:
        :param new_population:
        :return:
        """
        new_population = {}

        for species_size, species in zip(adjusted_species_sizes, remaining_species):

            assert (species_size > 0)

            # List of old species members
            old_species_members = list(species.members.values())
            # Reset the members for the current species
            species.members = {}
            # Save the species in the species set object
            species_set.species[species.key] = species

            # Sort the members into the descending fitness
            old_species_members.sort(reverse=True, key=lambda x: x.fitness)

            # Double check that it is descending
            if len(old_species_members) > 1:
                assert (old_species_members[0].fitness >= old_species_members[1].fitness)

            # If we have specified a number of genomes to carry over, carry them over to the new population
            num_genomes_without_crossover = int(round(species_size * self.config.chance_for_mutation_without_crossover))
            if num_genomes_without_crossover > 0:

                for member in old_species_members[:num_genomes_without_crossover]:

                    # Check if we should carry over a member un-mutated or not
                    if not self.config.keep_unmutated_top_percentage:
                        child = copy.deepcopy(member)

                        child.mutate(reproduction_instance=self,
                                     innovation_tracker=self.innovation_tracker, config=self.config)

                        if not child.check_connection_enabled_amount() and not child.check_num_paths(
                                only_add_enabled_connections=True):
                            raise Exception('This child has no enabled connections')

                        new_population[child.key] = child
                        self.ancestors[child.key] = ()
                        # new_population[member.key] = member
                        species_size -= 1
                        assert (species_size >= 0)
                    else:
                        # Else we just add the current member to the new population
                        new_population[member.key] = member
                        species_size -= 1
                        assert (species_size >= 0)

            # If there are no more genomes for the current species, then restart the loop for the next species
            if species_size <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            reproduction_cutoff = int(math.ceil((1 - self.config.chance_for_mutation_without_crossover) *
                                                len(old_species_members)))

            # Need at least two parents no matter what the previous result
            reproduction_cutoff = max(reproduction_cutoff, 2)
            old_species_members = old_species_members[:reproduction_cutoff]

            # Randomly choose parents and choose whilst there can still be additional genomes for the given species
            while species_size > 0:
                species_size -= 1

                # TODO: If you don't allow them to mate with themselves then it's a problem because if the species previous
                # TODO: size is 1, then how can you do with or without crossover?
                parent_1 = copy.deepcopy(random.choice(old_species_members))
                parent_2 = copy.deepcopy(random.choice(old_species_members))

                # Has to be a deep copy otherwise the connections which are crossed over are also modified if mutation
                # occurs on the child.
                parent_1 = copy.deepcopy(parent_1)
                parent_2 = copy.deepcopy(parent_2)

                self.genome_indexer += 1
                genome_id = self.genome_indexer

                child = Genome(key=genome_id)
                # TODO: Save the parent_1 and parent_2 mutation history as well as what connections they had
                # Create the genome from the parents
                num_connections_enabled = child.crossover(genome_1=parent_1, genome_2=parent_2, config=self.config)

                # If there are no connections enabled we forget about this child and don't add it to the existing
                # population
                if num_connections_enabled:
                    child.mutate(reproduction_instance=self,
                                 innovation_tracker=self.innovation_tracker, config=self.config)

                    if not child.check_connection_enabled_amount() and not child.check_num_paths(
                            only_add_enabled_connections=True):
                        raise Exception('This child has no enabled connections')

                    new_population[child.key] = child
                    self.ancestors[child.key] = (parent_1.key, parent_2.key)
                else:
                    assert num_connections_enabled == 0
                    species_size += 1
                    self.genome_indexer -= 1

        return new_population

    def reproduce(self, species_set, population_size, generation):
        """
        Handles reproduction of a population
        :param generation: Which generation number it is
        :param species_set: The SpeciesSet instance which keeps track of species
        :param population_size: The population size
        :return: A new population
        """
        # Check it is a class instance
        assert (isinstance(species_set, SpeciesSet))

        all_fitnesses, remaining_species = self.get_non_stagnant_species(species_set=species_set, generation=generation)

        adjusted_species_sizes = self.get_adjusted_species_sizes(all_fitnesses=all_fitnesses,
                                                                 population_size=population_size,
                                                                 remaining_species=remaining_species)

        # Set the species dict to an empty one for now as the new species will be configured later
        species_set.species = {}

        # Keeps track of the new population (key, object)
        new_population = self.get_new_population(adjusted_species_sizes=adjusted_species_sizes, species_set=species_set,
                                                 remaining_species=remaining_species)

        return new_population
