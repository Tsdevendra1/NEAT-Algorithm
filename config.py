class Config:
    """
    Population
    """

    population_size = 15

    # TODO: Ensure each of these are used somewhere in the code
    """
    Compatibility distance
    """
    # The coefficients used for calculating the compatibility distance between two genomes
    excess_coefficient = 1
    disjoint_coefficient = 1
    # This is for when the genes are the same so they check the similarity of the weights
    matching_genes_coefficient = 0.4

    # How close they have to be to be considered in the same species
    compatibility_threshold = 3

    """
    Mutation
    """
    # Whether only one type of mutation can happen at any time.
    single_mutation_only = False

    # Weight changes
    weight_mutation_chance = 0.8
    weight_mutation_perturbe_chance = 0.9
    weight_mutation_reset_connection_chance = 0.1

    weight_mutation_perturbe_chance_backprop = 0.9
    weight_mutation_reset_connection_chance_backprop = 0.1
    weight_mutation_reset_all_connections_chance_backprop = 0.1

    # Standard deviation for the distribution for which we pick the pertubation value from
    weight_mutation_sigma = 0.5
    # Mean for the distribution for which we pick the pertubation value from
    weight_mutation_mean = 0.0

    # Backprop versions
    weight_mutation_sigma_backprop = 1
    weight_mutation_mean_backprop = 0.0

    # This is the chance a gene is disabled if it was disabled in either parent
    change_to_disable_gene_if_either_parent_disabled = 0.75

    chance_for_mutation_without_crossover = 0.25

    inter_species_mating_rate = 0.001


    add_node_mutation_chance = 0.4
    add_connection_mutation_chance = 0.5
    remove_node_mutation_chance = 0.3
    remove_connection_mutation_chance = 0.2

    ## These chances are used when we're performing mutation whilst using backprop optimisation
    add_node_mutation_chance_backprop = 0.7
    add_connection_mutation_chance_backprop = 0.5
    remove_node_mutation_chance_backprop = 0.05
    remove_connection_mutation_chance_backprop = 0.05
    reset_all_weights_mutation_chance_backprop = 0.05
    # add_node_mutation_chance_backprop = 0.4
    # add_connection_mutation_chance_backprop = 0.7
    # remove_node_mutation_chance_backprop = 0.05
    # remove_connection_mutation_chance_backprop = 0.05
    # reset_all_weights_mutation_chance_backprop = 0.05

    ## OLD VALUES FOR WHEN WE WERE JUST USING GENETIC ALGORITHM
    # add_node_mutation_chance = 0.03
    # add_connection_mutation_chance = 0.05
    # remove_node_mutation_chance = 0.01
    # remove_connection_mutation_chance = 0.01

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
    # TODO: Change back to default value of 2
    min_species_size = 0

    """
    Survival
    """
    # Percentage of the population which carries on un-changed(?)
    survival_threshold = 0.2
    # This means that a certain percentage of the top elite genomes will carry over to the next population un changed
    keep_unmutated_top_percentage = True  # (Default should be False)
