class Config:
    # TODO: Ensure each of these are used somewhere in the code
    """
    Compatibility distance
    """
    # The coefficients used for calculating the compatibility distance between two genomes
    excess_coefficient = 1
    disjoint_coefficient = 1
    matching_genes_coefficient = 0.4

    """
    Mutation
    """

    # Weight changes
    weight_mutation_chance = 0.8
    weight_uniform_mutation_chance = 0.9
    weight_random_mutation_chance = 0.1
    # A range to choose from for the weight pertubation amount
    weight_range_low = 0.1
    weight_range_high = 0.8

    # This is the chance a gene is disabled if it was disabled in either parent
    chance_to_disable = 0.75

    # TODO: Does this mean that there is 75% chance for crossover or is it a fixed 75% (see the paper page 112)
    chance_for_mutation_without_crossover = 0.25

    inter_species_mating_rate = 0.001

    add_node_mutation_chance = 0.03
    add_connection_mutation_chance = 0.3
    # TODO: Find correct chance for these values
    remove_node_mutation_chance = 0.01
    remove_connection_mutation_chance = 0.05

    """
    Speciation
    """
    # Parameters used when check stagnation
    # Allowable number of generations before considered stagnant
    max_stagnation_generations = 15
    # Min number of species required before throwing out due to stagnation
    num_species_min = 2

    """
    Reproduction
    """
    # Minimum species size
    min_species_size = 2
    # How may of the best performing genomes are carried over for each species
    num_best_genome_carry_over = 2

    """
    Survival
    """
    # Percentage of the population which carries on un-changed(?)
    survival_threshold = 0.2
