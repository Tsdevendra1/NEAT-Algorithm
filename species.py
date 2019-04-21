import numpy as np


class Species:
    def __init__(self, key, generation):
        self.key = key
        # Which generation the species was created
        self.created = generation
        # Keeps track of what generation the fitness improved
        self.last_improved = generation
        self.representative = None
        # All members of the species
        self.members = {}
        self.fitness = None
        self.adjusted_fitness = None
        # History of the fitness for the species
        self.fitness_history = []

    def update(self, representative, members):
        self.representative = representative
        self.members = members


class SpeciesSet:

    def __init__(self, config, generation_tracker):
        self.config = config
        self.species_indexer = 0
        self.species = {}
        # For each genome if you index the dict it will return which species it is a part of
        self.genome_species = {}
        self.generation_tracker = generation_tracker

    @staticmethod
    def species_fitness_function(species_members, function_type):
        """
        Finds the fitness for a species. For now all it doesn is find the mean fitness of the species.
        :param species_members: The members of the species
        :param function_type: What type of function you want to apply, e.g. mean, max, min, median
        :return: The fitness value for the species
        """
        # TODO: Allow max, min, median fitness function types
        if function_type not in {'mean', 'max', 'median', 'min'}:
            raise Exception('Invalid function type specified for species fitness function')

        species_member_fitnesses = [member.fitness for member in species_members.values()]

        if function_type == 'mean':
            return np.mean(species_member_fitnesses)

    def calculate_compatibility_distance(self, species_representative, genome, generation_tracker=None):
        compatibility_distance_1 = species_representative.compute_compatibility_distance(other_genome=genome,
                                                                                         config=self.config, generation_tracker=generation_tracker)
        compatibility_distance_2 = genome.compute_compatibility_distance(other_genome=species_representative,
                                                                         config=self.config)

        # There's no reason for this to be different depending on who you choose to be the other genome
        if round(compatibility_distance_1, 3) != round(compatibility_distance_2, 3):
            raise Exception('There is an error in how compatibility distance is calculated')

        return compatibility_distance_1

    def find_new_species_representative(self, unspeciated, population, dict_of_compatibility_distances,
                                        new_representatives, new_members):
        """
        :param unspeciated: Set of genome_id's which haven't been assigned a species
        :param population: A dict of (genome_id, genome_class) for the population
        :param dict_of_compatibility_distances: An empty dict to store the distance between different genomes
        :param new_representatives: A dict to save the new representative for a species
        :param new_members: A dict to save the new members for each of the species
        """
        # For each species we find the new representative
        for species_id, species_object in self.species.items():
            candidates = []
            for genome_id in unspeciated:
                genome = population[genome_id]
                compatibility_distance = self.calculate_compatibility_distance(
                    species_representative=species_object.representative,
                    genome=genome)
                dict_of_compatibility_distances[(species_object.representative, genome)] = compatibility_distance
                candidates.append((compatibility_distance, genome))

            _, new_rep = min(candidates, key=lambda x: x[0])
            # Set the new representative for the species for the genome with the lowest distance
            new_rep_id = new_rep.key
            new_representatives[species_id] = new_rep_id
            new_members[species_id] = [new_rep_id]
            unspeciated.remove(new_rep_id)

    def find_species_members(self, unspeciated, population, dict_of_compatibility_distances,
                             new_representatives, new_members, compatibility_threshold, generation_tracker):
        while unspeciated:
            genome_id = unspeciated.pop()
            genome = population[genome_id]

            candidates = []

            # Keeps track of the distances with each representative
            compatibility_distances_dict = {}
            representative_genomes_list = []

            for species_id, representative_id in new_representatives.items():
                representative_genome = population[representative_id]
                compatibility_distance = self.calculate_compatibility_distance(
                    species_representative=representative_genome, genome=genome, generation_tracker=generation_tracker)

                compatibility_distances_dict[representative_genome] = compatibility_distance
                representative_genomes_list.append(representative_genome)

                dict_of_compatibility_distances[(representative_genome, genome)] = compatibility_distance
                if compatibility_distance < compatibility_threshold:
                    candidates.append((compatibility_distance, species_id))

            # Check to see if any of the representatives are below the threshold
            no_distances_less_than_threshold = True
            for distance in compatibility_distances_dict.values():
                if distance < compatibility_threshold:
                    no_distances_less_than_threshold = False
                    break

            if no_distances_less_than_threshold:
                # TODO: Remove this if not necessary
                random = 3

            if candidates:
                _, species_id = min(candidates, key=lambda x: x[0])
                new_members[species_id].append(genome_id)

            # We have to create a new species for the genome since it's not compatible
            else:
                # increment for a new species
                self.species_indexer += 1

                species_id = self.species_indexer
                new_representatives[species_id] = genome_id
                new_members[species_id] = [genome_id]

    def save_species_info(self, new_representatives, generation, new_members, population):
        """
        saves the species to a species object and then within the species set instance
        :param new_representatives: The representatives for the new species
        :param generation: Which generation number it is
        :param new_members: The members for each species
        :param population: A dict of (key: genome instance)
        """
        # For each genome_id keeps track of which species_id it belongs to
        self.genome_species = {}

        for species_id, representative_id in new_representatives.items():
            species_object = self.species.get(species_id)

            if species_object is None:
                species_object = Species(key=species_id, generation=generation)
                self.species[species_id] = species_object

            members = new_members[species_id]

            for genome_id in members:
                self.genome_species[genome_id] = species_id

            members_dict = dict((genome_id, population[genome_id]) for genome_id in members)
            species_object.update(representative=population[representative_id], members=members_dict)

    def speciate(self, population, compatibility_threshold, generation, generation_tracker=None):
        """

        :param generation: Which generation number it is
        :param compatibility_threshold:
        :param population: A dict of (genome_id, genome_class)
        :param generation_tracker: Tracks attributes for the current generation
        :return:
        """
        unspeciated = set(population)
        new_representatives = {}
        new_members = {}
        dict_of_compatibility_distances = {}

        self.find_new_species_representative(unspeciated=unspeciated,
                                             dict_of_compatibility_distances=dict_of_compatibility_distances,
                                             new_members=new_members, new_representatives=new_representatives,
                                             population=population)

        self.find_species_members(unspeciated=unspeciated,
                                  dict_of_compatibility_distances=dict_of_compatibility_distances,
                                  new_members=new_members, new_representatives=new_representatives,
                                  compatibility_threshold=compatibility_threshold, population=population, generation_tracker=generation_tracker)

        self.save_species_info(new_representatives=new_representatives, new_members=new_members, population=population,
                               generation=generation)

        # Mean compatability distance
        self.generation_tracker.mean_compatibility_distance = np.mean(list(dict_of_compatibility_distances.values()))
        # Standard deviation
        self.generation_tracker.std_dev_compatibility_distance = np.std(list(dict_of_compatibility_distances.values()))
        self.generation_tracker.num_species = len(self.species)
