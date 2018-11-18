from species import SpeciesSet


class Reproduce:
    """
    Will handle reproduction of the genomes
    """

    def __init__(self, stagnation):
        self.global_innovation_number = 0
        self.stagnation = stagnation
        self.ancestors = {}

    def create_new_population(self):
        pass

    def reproduce(self, species_instance, population_size):
        # Check it is a class instance
        assert (isinstance(species_instance, SpeciesSet))


        # TODO: Keep track of the innovations that have occured
        current_generation_innovations = []

        pass
