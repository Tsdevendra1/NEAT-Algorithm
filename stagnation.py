import sys


class Stagnation:

    @staticmethod
    def update(species_set, generation, config):
        """
        Required interface method. Updates species fitness history information,
        checking for ones that have not improved in max_stagnation generations,
        and - unless it would result in the number of species dropping below the configured
        species_elitism parameter if they were removed,
        in which case the highest-fitness species are spared -
        returns a list with stagnant species marked for removal.
        """

        species_data = []
        for species_id, species in species_set.species.items():
            if species.fitness_history:
                # If there is fitness_history get the previous generation fitness
                prev_fitness = max(species.fitness_history)
            else:
                # Else just set it to the lowest possible value for now
                prev_fitness = -sys.float_info.max

            # Calculate the fitness for the species
            species.fitness = species_set.species_fitness_function(species_members=species.members,
                                                                   function_type='mean')

            # Keep track of historical fitness
            species.fitness_history.append(species.fitness)

            species.adjusted_fitness = None

            # Keep track of when the generation was last improved
            if prev_fitness is None or species.fitness > prev_fitness:
                species.last_improved = generation

            species_data.append((species_id, species))

        # Sort the species data into ascending fitness order.
        species_data.sort(key=lambda x: x[1].fitness)

        # Keeps track of which species are stagnant or not
        results = []
        # Keeps track of each species's fitness
        species_fitnesses = []
        num_non_stagnant = len(species_data)

        for index, (species_id, species) in enumerate(species_data):
            # Override stagnant state if marking this species as stagnant would
            # result in the total number of species dropping below the limit.
            # Because species are in ascending fitness order, less fit species
            # will be marked as stagnant first.
            stagnant_time = generation - species.last_improved
            is_stagnant = False

            if num_non_stagnant > config.num_species_min:
                # Check if the stagnation time for the species is greater than the max set in the config
                is_stagnant = (stagnant_time >= config.max_stagnation_generations)

            # This will ensure that whatever the value of species_min is will be allowed to continue un-stagnated even
            # if they are. Example: if self.config.num_species_min = 2, then as long as the index is the last two of the
            # length of species_data it will set is_stagnant to False
            if (len(species_data) - index) <= config.num_species_min:
                is_stagnant = False

            if is_stagnant:
                num_non_stagnant -= 1

            results.append((species_id, species, is_stagnant))
            species_fitnesses.append(species_fitnesses)

        return results
