from genome import Genome
from species import SpeciesSet
import random
import numpy as np
import math


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

    def create_new_population(self):
        pass

    def compute_adjusted_species_sizes(self, adjusted_species_fitnesses, previous_species_sizes, population_size,
                                       min_species_size):
        """
        Compute the number of offspring per species, proportinal to their fitnesses
        :param adjusted_species_fitnesses:
        :param previous_species_sizes:
        :param population_size:
        :param min_species_size:
        :return:
        """
        pass

    def get_non_stagnant_species(self, species_set, generation):
        """
        Checks which species are stagnant ant returns the ones which aren't
        :param species_set: The species set instance which stores all the species
        :return: A list of non stagnant species
        """
        # Keeps track of all the fitnesses for the genomes in the population
        all_fitnesses = []
        # Keeps track of the species which aren't stagnant
        remaining_species = []

        # (Id, species instance, boolean)
        for species_id, species, is_stagnant in self.stagnation.update(species_set=species_set, generation=generation):
            if is_stagnant:
                # TODO: What to do here??
                pass
            else:
                # Save all the fitness in the species that isn't stagnant
                all_fitnesses += [member.fitness for member in species.members.values()]
                remaining_species.append(species)

        # The case where there are no species left
        if not remaining_species:
            # TODO: Would this ever come here?
            raise Exception('There are no remaining species in the reproduce function')
            species_set.species = {}
            return {}

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
            mean_species_fitness = np.mean([member.fitness for member in species.members.values()])
            adjusted_fitness = (mean_species_fitness - min_genome_fitness) / fitness_range
            species.adjusted_fitness = adjusted_fitness

        adjusted_species_fitnesses = [species.adjusted_fitness for species in remaining_species]

        # Get a list of the amount of members in each of the remaining species
        previous_species_sizes = [len(species.member) for species in remaining_species]

        # The min species size is the maximum of either the minimum set species size of the number of the best genomes
        # are required to be carried over
        min_species_size = max(self.config.min_species_size, self.config.num_best_genome_carry_over)

        return self.compute_adjusted_species_sizes(
            adjusted_species_fitnesses=adjusted_species_fitnesses, min_species_size=min_species_size,
            previous_species_sizes=previous_species_sizes, population_size=population_size)

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
            species_size = max(species_size, self.config.num_best_genome_carry_over)

            assert (species_size > 0)

            # List of old species members
            old_species_members = list(species.members.values())
            # Keeps track of current species member
            species_members = {}
            # Save the species in the species set object
            species_set.species[species.key] = species

            # Sort the members into the descending fitness
            old_species_members.sort(reverse=True, key=lambda x: x.fitness)

            # Double check that it is descending
            if len(old_species_members) > 1:
                assert (old_species_members[0].fitness > old_species_members[1].fitness)

            # If we have specified a number of genomes to carry over, carry them over to the new population
            if self.config.num_best_genome_carry_over > 0:
                for member in old_species_members[:self.config.num_best_genome_carry_over]:
                    new_population[member.key] = member
                    species_size -= 1

            # If there are no more genomes for the current species, then restart the loop for the next species
            if species_size <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            reproduction_cutoff = int(math.ceil(self.config.survival_threshold *
                                                len(old_species_members)))

            # Need at least two parents no matter what the previous result
            reproduction_cutoff = max(reproduction_cutoff, 2)
            old_species_members = old_species_members[:reproduction_cutoff]

            # TODO: Keep track of the innovations that have occured
            # This dict will maintain which new connections have been added this generation as well as their innovation
            # number: (1(input),3(output)): 9
            current_generation_innovations = {}

            # Randomly choose parents and choose whilst there can still be additional genomes for the given species
            while species_size > 0:
                species_size -= 1

                # TODO: Can a genome mate with itself?
                parent_1 = random.choice(old_species_members)
                parent_2 = random.choice(old_species_members)

                self.genome_indexer += 1
                genome_id = self.genome_indexer

                child = Genome(key=genome_id)
                # Create the genome from the parents
                child.crossover(genome_1=parent_1, genome_2=parent_2)

                # Increment the global innovation number since a mutation will occur
                self.global_innovation_number += 1
                # TODO: Sort out saving of the innovation number things and returning
                child.mutate(new_innovation_number=, current_gen_innovations=, config=self.config)

                new_population[child.key] = child
                self.ancestors[child.key] = (parent_1.key, parent_2.key)

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
