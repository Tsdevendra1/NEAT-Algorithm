from deconstruct_genome_multiclass import DeconstructGenomeMultiClass
import copy

from genome_neural_network import GenomeNeuralNetwork
from graph_algorithm_mutliclass import GraphMultiClass
import itertools
import numpy as np
from gene import ConnectionGene, NodeGene
import random

from neural_network import create_data


class GenomeMultiClass:
    def __init__(self, key, connections=None, nodes=None):
        """
        :param key: Which genome number it is
        :param connections: A list of ConnectionGene instances
        :param nodes: A list of NodeGene Instance
        """

        # Unique identifier for a genome instance.
        self.key = key

        self.mutations_occured = []
        self.parents = None

        self.connections = {}
        self.nodes = {}

        # Fitness results.
        self.fitness = None

        # These attributes will be set when the genome is unpacked
        self.connection_matrices_per_layer = None
        self.no_activations_matrix_per_layer = None
        self.nodes_per_layer = None
        self.constant_weight_connections = None
        self.node_map = None
        self.layer_connections_dict = None
        self.updated_nodes = None
        self.layer_nodes = None
        self.node_layers = None
        self.num_layers_including_input = None
        self.last_dummy_related_to_connection = None

        if connections and nodes:
            # To be able to configure them they must be in a list
            assert isinstance(connections, list)
            assert isinstance(nodes, list)
            # Saves the genes in an appropriate format into the dictionaries above. (See the method for saved format)
            self.configure_genes(connections=connections, nodes=nodes)

            self.unpack_genome()

    def check_any_disabled_connections_in_path(self):
        """
        For a given path, if one of the connection genes is disabled, it will disable all the connection genes for that path
        :return:
        """
        connections_by_tuple = {}
        for connection in self.connections.values():
            connections_by_tuple[(connection.input_node, connection.output_node)] = connection

        _, all_paths = self.check_num_paths(only_add_enabled_connections=False, return_paths=True)
        connection_gene_enabled_tracker = {}
        all_tuples_for_paths = set()
        for node_paths in all_paths:
            for path in node_paths:
                any_disabled = False
                connection_gene_list = []
                for index in range(len(path) - 1):
                    connection_tuple = (path[index], path[index + 1])
                    connection_gene = connections_by_tuple[connection_tuple]
                    all_tuples_for_paths.add(connection_tuple)
                    connection_gene_list.append(connection_gene)
                    if not connection_gene.enabled:
                        any_disabled = True

                for connection in connection_gene_list:
                    if connection not in connection_gene_enabled_tracker:
                        if any_disabled:
                            connection_gene_enabled_tracker[connection] = [False]
                        else:
                            connection_gene_enabled_tracker[connection] = [True]
                    else:
                        if any_disabled:
                            connection_gene_enabled_tracker[connection].append(False)
                        else:
                            connection_gene_enabled_tracker[connection].append(True)

        # If the connection doesn't feature in any of the paths identified, we disable it as it isn't part of a path
        # at all
        for connection in self.connections.values():
            connection_tuple = (connection.input_node, connection.output_node)
            if connection_tuple not in all_tuples_for_paths:
                connection.enabled = False

        for connection, true_false_list in connection_gene_enabled_tracker.items():
            num_true = 0
            for boolean in true_false_list:
                if boolean:
                    num_true += 1
                    break
            if num_true == 0 and connection.enabled:
                connection.enabled = False
        return all_paths

    def unpack_genome(self):
        """
        Deconstructs the genome into a structure that can be used for the neural network it represents
        """

        # TODO: Look through connections. If there is a connection which isn't part of a path, disable it

        all_paths = self.check_any_disabled_connections_in_path()

        # Check that there are valid paths for the neural network
        num_source_to_output_paths = self.check_num_paths(only_add_enabled_connections=True)
        if not num_source_to_output_paths:
            return False
        if num_source_to_output_paths == 0:
            raise Exception('There is no valid path from any source to the output')

        # Unpack the genome and get the returning dictionary
        return_dict = DeconstructGenomeMultiClass.unpack_genome(genome=self, all_paths=all_paths)

        self.connection_matrices_per_layer = return_dict['connection_matrices']
        self.no_activations_matrix_per_layer = return_dict['bias_matrices']
        self.nodes_per_layer = return_dict['nodes_per_layer']
        self.constant_weight_connections = return_dict['constant_weight_connections']
        self.node_map = return_dict['node_map']
        self.layer_connections_dict = return_dict['layer_connections_dict']
        self.updated_nodes = return_dict['nodes']
        self.node_layers = return_dict['node_layers']
        self.layer_nodes = return_dict['layer_nodes']
        self.num_layers_including_input = max(self.layer_nodes)
        self.last_dummy_related_to_connection = return_dict['last_dummy_related_to_connection']

        self.check_output_nodes_on_last_layer()

        return True

    def check_output_nodes_on_last_layer(self):
        output_nodes = []
        for node in self.nodes.values():
            if node.node_type == 'output':
                output_nodes.append(node)

        for node in output_nodes:
            if self.node_layers[node.node_id] != max(self.layer_nodes):
                raise Exception('Output node was found to not be in the very last layer')

    def configure_genes(self, connections, nodes):
        """
        :param connections: A list of connection genes
        :param nodes: A list of node genes
        Sets up the self.connection and self.nodes dicts for the gene.
        """
        # (innovation_number, ConnectionGene class object) pairs for connection gene sets.
        for connection in connections:
            if connection.innovation_number in self.connections:
                raise KeyError('You do not have a unique innovation number for this connection')
            if connection.input_node == connection.output_node:
                raise ValueError('The input and output node cant be the same')
            self.connections[connection.innovation_number] = connection

        # (node_id, NodeGene class object) pairs for the node gene set
        for node in nodes:
            self.nodes[node.node_id] = node

    def check_connection_enabled_amount(self):
        """
        Checks how many enabled connections there are
        """
        num_enabled_connections = 0

        for connection in self.connections.values():
            if connection.enabled:
                num_enabled_connections += 1

        return num_enabled_connections

    def crossover(self, genome_1, genome_2, config):
        """
        :param genome_1:
        :param genome_2:
        :return:
        """
        assert isinstance(genome_1.fitness, (int, float))
        assert isinstance(genome_2.fitness, (int, float))

        if genome_1.fitness > genome_2.fitness:
            fittest_parent, second_parent = genome_1, genome_2
        # If they are equal it mean's its the same genome being used twice
        elif genome_1.fitness == genome_2.fitness:
            fittest_parent, second_parent = genome_1, genome_2
        else:
            fittest_parent, second_parent = genome_2, genome_1

        # Inherit connection genes
        for fittest_connection_gene in fittest_parent.connections.values():
            second_connection_gene = second_parent.connections.get(fittest_connection_gene.innovation_number)

            # If there is a disjoint or excess gene then it is inherited from the fittest parent only
            if second_connection_gene is None:
                self.connections[fittest_connection_gene.innovation_number] = fittest_connection_gene

            # If there is a second gene it means both genomes have the gene and hence we pick randomly for which one is
            # carried over
            else:

                connection_genes = [fittest_connection_gene, second_connection_gene]
                inherited_gene = copy.deepcopy(random.choice(connection_genes))
                # There is a chance for the gene to be disabled if ti is
                if not fittest_connection_gene.enabled or not second_connection_gene.enabled:
                    chance_to_disable_roll = np.random.uniform(low=0.0, high=1.0)
                    if chance_to_disable_roll < config.change_to_disable_gene_if_either_parent_disabled:
                        inherited_gene.enabled = False
                self.connections[inherited_gene.innovation_number] = inherited_gene

        # Inherit the node genes
        for fittest_node in fittest_parent.nodes.values():
            second_node = second_parent.nodes.get(fittest_node.node_id)
            assert (fittest_node.node_id not in self.nodes)
            # If the node isn't in the second genome it is inherited from fittest parent
            if second_node is None:
                self.nodes[fittest_node.node_id] = fittest_node
            # Choose randomly if both have the node
            else:
                node_genes = [fittest_node, second_node]
                inherited_node_gene = random.choice(node_genes)
                self.nodes[inherited_node_gene.node_id] = inherited_node_gene

        for genome in [genome_1, genome_2]:
            if not genome.check_connection_enabled_amount():
                raise Exception('One or both of the parents in crossover doesnt have valid connections')

        _, outputs_with_zero_paths = self.get_viable_nodes_to_delete(return_outputs_with_zero_path=True)
        # If no enabled paths for a certain output, enable them all
        if outputs_with_zero_paths:
            for connection in self.connections.values():
                for output_node in outputs_with_zero_paths:
                    if connection.output_node == output_node and not connection.enabled:
                        connection.enabled = True

        # TODO: What to do if you inherit genes and they are all disabled?
        num_enabled = 0
        copy_of_connections = copy.deepcopy(list(self.connections.values()))
        for connection in copy_of_connections:
            if connection.enabled:
                num_enabled += 1

        if num_enabled > 0:
            # Unpack the genome after we have configured it
            unpack_went_fine = self.unpack_genome()
            return unpack_went_fine

        return num_enabled

    def mutate(self, reproduction_instance, innovation_tracker, config, generation_tracker, backprop_mutation=False):
        """
        Will call one of the possible mutation abilities using a random() number generated
        :return:
        """

        self_copy_for_debugging = copy.deepcopy(self)

        if config.single_mutation_only and not backprop_mutation:
            # The boundaries set for different mutation types don't work if the mutation chances add up to more than 1
            assert (
                    config.add_connection_mutation_chance + config.add_node_mutation_chance + config.remove_connection_mutation_chance + config.remove_node_mutation_chance
                    <= 1)
            mutation_roll = np.random.uniform(low=0.0, high=1.0)

            # Boundaries for different rolls (These are all higher end of the boundary)
            add_connection_boundary = config.add_connection_mutation_chance
            add_node_boundary = add_connection_boundary + config.add_node_mutation_chance
            remove_node_boundary = add_node_boundary + config.remove_node_mutation_chance
            remove_connection_boundary = remove_node_boundary + config.remove_connection_mutation_chance

            add_connection_condition = 0 < mutation_roll <= add_connection_boundary
            add_node_condition = add_connection_boundary < mutation_roll <= add_node_boundary
            remove_node_condition = add_node_boundary < mutation_roll <= remove_node_boundary
            remove_connection_condition = remove_node_boundary < mutation_roll <= remove_connection_boundary

        elif not backprop_mutation:
            # The rolls to see if each mutation occurs
            add_connection_roll = np.random.uniform(low=0.0, high=1.0)
            add_node_roll = np.random.uniform(low=0.0, high=1.0)
            remove_node_roll = np.random.uniform(low=0.0, high=1.0)
            remove_connection_roll = np.random.uniform(low=0.0, high=1.0)

            add_connection_condition = add_connection_roll <= config.add_connection_mutation_chance
            add_node_condition = add_node_roll <= config.add_node_mutation_chance
            remove_node_condition = remove_node_roll <= config.remove_node_mutation_chance
            remove_connection_condition = remove_connection_roll <= config.remove_connection_mutation_chance
        elif backprop_mutation and config.single_mutation_only:
            mutation_roll = np.random.uniform(low=0.0, high=1.0)

            # Boundaries for different rolls (These are all higher end of the boundary)
            add_connection_boundary = config.add_connection_mutation_chance_backprop
            add_node_boundary = add_connection_boundary + config.add_node_mutation_chance_backprop
            remove_node_boundary = add_node_boundary + config.remove_node_mutation_chance_backprop
            remove_connection_boundary = remove_node_boundary + config.remove_connection_mutation_chance_backprop

            add_connection_condition = 0 < mutation_roll <= add_connection_boundary
            add_node_condition = add_connection_boundary < mutation_roll <= add_node_boundary
            remove_node_condition = add_node_boundary < mutation_roll <= remove_node_boundary
            remove_connection_condition = remove_node_boundary < mutation_roll <= remove_connection_boundary

        elif backprop_mutation:
            # The rolls to see if each mutation occurs
            add_connection_roll = np.random.uniform(low=0.0, high=1.0)
            add_node_roll = np.random.uniform(low=0.0, high=1.0)
            remove_node_roll = np.random.uniform(low=0.0, high=1.0)
            remove_connection_roll = np.random.uniform(low=0.0, high=1.0)

            add_connection_condition = add_connection_roll <= config.add_connection_mutation_chance_backprop
            add_node_condition = add_node_roll <= config.add_node_mutation_chance_backprop
            remove_node_condition = remove_node_roll <= config.remove_node_mutation_chance_backprop
            remove_connection_condition = remove_connection_roll <= config.remove_connection_mutation_chance_backprop
        else:
            raise Exception('Unknown mutation type detected in genome.py, mutation function')

            # Add connection
        if add_connection_condition:
            self.add_connection(reproduction_instance=reproduction_instance,
                                innovation_tracker=innovation_tracker)
            # Unpack the genome after whatever mutation has occured
            self.mutations_occured.append('Add Connection')
            self.unpack_genome()
            generation_tracker.num_generation_add_connection += 1

        # Add node
        if add_node_condition:
            self.add_node(reproduction_instance=reproduction_instance,
                          innovation_tracker=innovation_tracker)
            self.mutations_occured.append('Add Node')
            # Unpack the genome after whatever mutation has occured
            self.unpack_genome()
            generation_tracker.num_generation_add_node += 1

        # Remove node
        if remove_node_condition:
            # I use this to see what the genome was like before I deleted the node
            self_copy_for_debugging_node = copy.deepcopy(self)
            self.remove_node()
            self.mutations_occured.append('Remove Node')
            # Unpack the genome after whatever mutation has occured
            self.unpack_genome()
            generation_tracker.num_generation_delete_node += 1

        # Remove connection
        if remove_connection_condition:
            # I use this to see what the genome was like before I deleted the connection
            self_copy_for_debugging_connection = copy.deepcopy(self)
            self.remove_connection()
            self.mutations_occured.append('Remove Connection')
            # Unpack the genome after whatever mutation has occured
            self.unpack_genome()
            generation_tracker.num_generation_delete_connection += 1

        # Mutate weight (This is independent of all other mutations)
        if np.random.uniform(low=0.0, high=1.0) < config.weight_mutation_chance:
            self.mutate_weight(config=config, generation_tracker=generation_tracker,
                               backprop_mutation=backprop_mutation)
            generation_tracker.num_generation_weight_mutations += 1

    def clean_combinations(self, possible_combinations):
        """
        Cleans the possible combinations so that only combinations that are possible are available to be chosen from
        :param possible_combinations: A iterable of tuples for the possible combinations between nodes
        :return: A list of viable combinations
        """
        cleaned_combinations = []
        for combination in possible_combinations:
            input_node = combination[0]
            output_node = combination[1]

            # Skip the loop if it's connecting two source nodes
            if self.nodes[input_node].node_type == 'source' and self.nodes[output_node].node_type == 'source' or \
                    self.nodes[input_node].node_type == 'output':
                continue

            input_node_type = self.nodes[input_node].node_type

            try:
                input_node_layer = self.node_layers[input_node]
            except KeyError:
                raise Exception
            output_node_layer = self.node_layers[output_node]

            # As long as the node type isn't a source then the connecting node can be on the same layer or greater
            if input_node_type != 'source' and output_node_layer >= input_node_layer:
                cleaned_combinations.append(combination)
            # If the input node type is a source then the connecting node can't be on the same layer
            elif input_node_type == 'source' and output_node_layer > input_node_layer:
                cleaned_combinations.append(combination)

        return cleaned_combinations

    def get_active_nodes(self):
        """
        :return: A list of nodes which are attached to active connections. Otherwise we assume the node is switched off
        """
        active_nodes = set()
        for connection in self.connections.values():
            if connection.enabled:
                active_nodes.add(connection.input_node)
                active_nodes.add(connection.output_node)

        return list(active_nodes)

    def add_connection(self, reproduction_instance, innovation_tracker):
        """
        Add a random connection
        :param innovation_tracker: The innovations that have happened in the past
        :param reproduction_instance: An instance of the reproduction class so we can access the global innovation number

        """

        # Keeps a list of nodes which can't be chosen from to be the start node of the new connection
        viable_start_nodes = []

        # Any node that isn't the output node is a viable start_node
        for node_id, node in self.nodes.items():
            if node.node_type != 'output':
                viable_start_nodes.append(node_id)

        possible_combinations = itertools.combinations(self.get_active_nodes(), 2)
        cleaned_possible_combinations = self.clean_combinations(possible_combinations=possible_combinations)

        # Get all the existing combinations
        already_existing_connections = set()
        for connection in self.connections.values():
            already_existing_connections.add((connection.input_node, connection.output_node))

        possible_new_connections = []
        for connection in cleaned_possible_combinations:
            # If it's not an existing combination then we can choose from it randomly
            if connection not in already_existing_connections:
                possible_new_connections.append(connection)

        if possible_new_connections:
            # Pick randomly from the possible new choices
            new_connection = random.choice(possible_new_connections)

            if new_connection in innovation_tracker:
                # If the connection was already made then use whatever the innovation number that was assigned to
                # that connection
                new_connection_gene = ConnectionGene(input_node=new_connection[0], output_node=new_connection[1],
                                                     innovation_number=innovation_tracker[new_connection],
                                                     weight=np.random.randn())

            else:
                reproduction_instance.global_innovation_number += 1
                new_connection_gene = ConnectionGene(input_node=new_connection[0], output_node=new_connection[1],
                                                     innovation_number=reproduction_instance.global_innovation_number,
                                                     weight=np.random.randn())

                # Save the connection as a current gen innovation
                innovation_tracker[new_connection] = reproduction_instance.global_innovation_number

            # Add the connection the the genome
            self.connections[new_connection_gene.innovation_number] = new_connection_gene

            return new_connection_gene

    def keep_node(self, node):
        """
        A node is suitable to delete if it it doesn't have both an input connection and output connection
        :param node: The node to check if it can be deleted
        :return: Whether a node should be kept or not
        """
        # Keep track if the node has any other input or output connections once the connection we have removed is
        # deleted
        has_input_connection = False
        has_output_connection = False
        connections_list = []
        for connection in self.connections.values():

            # Check if there is an actual input into the node
            if connection.output_node == node:
                has_input_connection = True
                connections_list.append(connection)

            # check if there is an output connection for the node
            if connection.input_node == node:
                has_output_connection = True
                connections_list.append(connection)

        return has_input_connection and has_output_connection, connections_list

    def check_which_connections_removable(self):
        """
        :return:
        Returns a list of connections which can be removed
        """
        num_source_to_output_paths, all_paths = self.check_num_paths(only_add_enabled_connections=True,
                                                                     return_paths=True)

        connection_count = {}
        for node_paths in all_paths:
            for path in node_paths:
                for index in range(len(path) - 1):
                    connection_tuple = (path[index], path[index + 1])
                    if connection_tuple not in connection_count:
                        connection_count[connection_tuple] = 1
                    else:
                        connection_count[connection_tuple] += 1

        connections_by_tuple = {}
        for connection in self.connections.values():
            connections_by_tuple[(connection.input_node, connection.output_node)] = connection

        viable_nodes_to_delete, outputs_with_one_path = self.get_viable_nodes_to_delete(
            return_outputs_with_one_path=True, return_outputs_with_zero_path=True)

        choice_list = []
        for connection, connection_amount in connection_count.items():
            output_with_one_path_paths_list = []
            for output in outputs_with_one_path:
                for node_paths in all_paths:
                    for path in node_paths:
                        if output in path:
                            output_with_one_path_paths_list.append(path)

            node_in_critical_path = False
            for path in output_with_one_path_paths_list:
                # Minus one because we use +1 in index referencing below to get the next one along
                for index in range(len(path) - 1):
                    tuple_connnection_input = path[index]
                    tuple_connnection_output = path[index + 1]
                    connection_tuple_in_path = (tuple_connnection_input, tuple_connnection_output)
                    if connection == connection_tuple_in_path:
                        node_in_critical_path = True
                        break
                if node_in_critical_path:
                    break

            if connection_amount != num_source_to_output_paths and not node_in_critical_path:
                choice_list.append(connections_by_tuple[connection])

        return choice_list

    def check_num_paths(self, only_add_enabled_connections, return_paths=False, return_graph_layer_nodes=False):
        source_nodes = []
        output_nodes = []

        # for node in self.nodes.values():
        #     if node.node_type == 'source':
        #         source_nodes.append(node)
        #     elif node.node_type == 'output':
        #         output_nodes.append(node)

        # Getting input and output nodes this way is better because then it only includes input nodes which have a connection
        for connection in self.connections.values():
            input_node = self.nodes[connection.input_node]
            output_node = self.nodes[connection.output_node]
            if input_node.node_type == 'source':
                source_nodes.append(input_node)
            if output_node.node_type == 'output':
                output_nodes.append(output_node)

        # Remove duplicates
        source_nodes = list(set(source_nodes))
        output_nodes = list(set(output_nodes))

        graph = GraphMultiClass()
        # Add the connections to the graph
        for connection in self.connections.values():
            if only_add_enabled_connections:
                if connection.enabled:
                    graph.add_edge(start_node=connection.input_node, end_node=connection.output_node)
            else:
                graph.add_edge(start_node=connection.input_node, end_node=connection.output_node)

        # Only keep the unique nodes to there are no duplicates in the list
        source_nodes = list(set(source_nodes))
        output_nodes = list(set(output_nodes))

        if not output_nodes:
            return False

        # Keeps track of how many paths there are from the source to the input's
        num_source_to_output_paths = 0
        all_paths = []
        for output_node in output_nodes:
            for node in source_nodes:
                if return_paths:
                    path_amount, paths = graph.count_paths(start_node=node.node_id, end_node=output_node.node_id,
                                                           return_paths=True)
                    num_source_to_output_paths += path_amount
                    all_paths.append(paths)
                else:
                    num_source_to_output_paths += graph.count_paths(start_node=node.node_id,
                                                                    end_node=output_node.node_id)

        if return_paths:
            return num_source_to_output_paths, all_paths
        else:
            if return_graph_layer_nodes and only_add_enabled_connections:
                return num_source_to_output_paths, graph.max_layer_for_node
            else:
                return num_source_to_output_paths

    def remove_connection(self):
        """
        Removes a random existing connection form the genome
        """

        num_source_to_output_paths = self.check_num_paths(only_add_enabled_connections=True)

        # If there is only one path from the source to the output we shouldn't delete any of the connections since
        # it would make it an invalid network
        if num_source_to_output_paths > 1:
            choice_list = self.check_which_connections_removable()
            if choice_list:
                connection_to_remove = random.choice(choice_list)

                input_node = connection_to_remove.input_node
                input_node_type = self.nodes[connection_to_remove.input_node].node_type
                output_node = connection_to_remove.output_node
                output_node_type = self.nodes[connection_to_remove.output_node].node_type

                # Delete the connection
                del self.connections[connection_to_remove.innovation_number]

                # Check if the input_node can be kept, delete if not. This is only the case if it is not a source/output
                # node. Because we want to allow them to be able to have connections in the future
                keep_input, input_connections = self.keep_node(node=input_node)
                if input_node_type != 'source' and input_node_type != 'output' and not keep_input:
                    del self.nodes[input_node]
                    # Delete connections related to input node
                    for connection in input_connections:
                        del self.connections[connection.innovation_number]

                # Check if the output_node can be kept, delete if not. This is only the case if it is not a source node.
                # Because we want to allow source nodes to have connections in the future
                keep_output, output_connections = self.keep_node(node=output_node)
                if output_node_type != 'source' and output_node_type != 'output' and not keep_output:
                    del self.nodes[output_node]
                    # Delete connections related to output node
                    for connection in output_connections:
                        del self.connections[connection.innovation_number]

                print('CONNECTION BEING DELETED: ', connection_to_remove)
                return connection_to_remove

    def add_node(self, reproduction_instance, innovation_tracker):
        """
        Add a node between two existing nodes
        """

        # Create the new node
        new_node = NodeGene(node_id=(max(self.nodes.keys()) + 1), node_type='hidden', bias=1)

        # The connection where the node will be added
        connections_list = list(self.connections.values())
        # Remove any connections which aren't enabled as a candidate where a node can be added
        for connection in connections_list:
            if not connection.enabled:
                connections_list.remove(connection)

        connection_to_add_node = random.choice(connections_list)
        # Disable the connection since it will be replaced
        connection_to_add_node.enabled = False

        input_node = connection_to_add_node.input_node
        output_node = connection_to_add_node.output_node

        # Create new connection gene. Which has a weight of 1
        first_combination = (input_node, new_node.node_id)
        if first_combination in innovation_tracker:
            first_new_connection = ConnectionGene(input_node=input_node, output_node=new_node.node_id,
                                                  innovation_number=innovation_tracker[first_combination],
                                                  weight=1)
        else:
            # Increment since there is a new innovation
            reproduction_instance.global_innovation_number += 1
            first_new_connection = ConnectionGene(input_node=input_node, output_node=new_node.node_id,
                                                  innovation_number=reproduction_instance.global_innovation_number,
                                                  weight=1)
            # Save the innovation since it's new
            innovation_tracker[first_combination] = reproduction_instance.global_innovation_number
        second_combination = (new_node.node_id, output_node)

        # The second connection keeps the weight of the connection it replaced
        if second_combination in innovation_tracker:
            second_new_connection = ConnectionGene(input_node=new_node.node_id, output_node=output_node,
                                                   innovation_number=innovation_tracker[second_combination],
                                                   weight=connection_to_add_node.weight)
        else:
            # Increment since there is a new innovation
            reproduction_instance.global_innovation_number += 1
            second_new_connection = ConnectionGene(input_node=new_node.node_id, output_node=output_node,
                                                   innovation_number=reproduction_instance.global_innovation_number,
                                                   weight=connection_to_add_node.weight)
            # save the innovation if it doesn't already exist
            innovation_tracker[second_combination] = reproduction_instance.global_innovation_number

        # Add the new node and connections
        self.nodes[new_node.node_id] = new_node
        self.connections[first_new_connection.innovation_number] = first_new_connection
        self.connections[second_new_connection.innovation_number] = second_new_connection

        return first_new_connection, second_new_connection

    def get_viable_nodes_to_delete(self, return_outputs_with_one_path=False, return_outputs_with_zero_path=False):
        """
        :return: A list of nodes which are available to be deleted
        """
        viable_nodes_to_be_delete = []
        # connections_able_to_remove = self.check_which_connections_removable()
        num_source_to_output_paths, all_paths = self.check_num_paths(only_add_enabled_connections=True,
                                                                     return_paths=True)

        # How many times a node comes up in all paths
        node_count = {}
        for node_paths in all_paths:
            for path in node_paths:
                for node in path:
                    if node not in node_count:
                        node_count[node] = 1
                    else:
                        node_count[node] += 1

        output_node_ids = set()
        output_node_count = {}
        # Remove any output nodes for contention for deletion
        for node in self.nodes.values():
            # if node.node_type == 'output' and node.node_id in node_count:
            #     del node_count[node.node_id]
            if node.node_type == 'output':
                output_node_ids.add(node.node_id)
                if node.node_id in node_count:
                    output_node_count[node.node_id] = node_count[node.node_id]
                    del node_count[node.node_id]

        if return_outputs_with_zero_path and return_outputs_with_one_path:
            output_nodes_with_one_path = [node for node, node_amount in output_node_count.items() if
                                          node_amount == 0 or node_amount == 1]
        elif return_outputs_with_one_path:
            output_nodes_with_one_path = [node for node, node_amount in output_node_count.items() if
                                          node_amount == 1]
        elif return_outputs_with_zero_path:
            output_nodes_with_one_path = [node for node, node_amount in output_node_count.items() if
                                          node_amount == 0]
        else:
            output_nodes_with_one_path = [node for node, node_amount in output_node_count.items() if
                                          node_amount == 0 or node_amount == 1]

        # Need to catch output_nodes which don't have ANY connections
        for node_id in output_node_ids:
            if node_id not in output_node_count and node_id not in output_nodes_with_one_path:
                output_nodes_with_one_path.append(node_id)

        # Delete node from node_count if it doesn't have a path
        for output_node in output_nodes_with_one_path:
            for connection in self.connections.values():
                # If it is the only connection to the output_node we have to remove it from the pool of available
                # nodes to be deleted
                if connection.output_node == output_node and node_count.get(connection.input_node):
                    del node_count[connection.input_node]

        # Add viable nodes that can be deleted
        for node, node_amount in node_count.items():
            if node_amount != num_source_to_output_paths and node not in output_node_ids:
                viable_nodes_to_be_delete.append(node)

        if return_outputs_with_one_path or return_outputs_with_zero_path:
            return viable_nodes_to_be_delete, output_nodes_with_one_path
        return viable_nodes_to_be_delete

    def remove_node(self, node_to_remove=None):
        """

        :param node_to_remove: For debug purposes if you want to remove a specific node
        :return:
        """
        num_source_to_output_paths = self.check_num_paths(only_add_enabled_connections=True)

        # Otherwise if you delete a node and it only has one path, you make it an invalid network as there is not path
        # from the source to the output
        if num_source_to_output_paths > 1:

            viable_nodes_to_be_delete = self.get_viable_nodes_to_delete()

            # Remove duplicates
            viable_nodes_to_be_delete = list(set(viable_nodes_to_be_delete))

            # If there are any viable nodes to delete
            if viable_nodes_to_be_delete:

                # Randomly choose node to delete
                node_to_delete = random.choice(viable_nodes_to_be_delete) if not node_to_remove else node_to_remove

                # The node to be deleted will have connections which also need to be deleted
                connections_to_delete = []

                for connection in self.connections.values():
                    if connection.input_node == node_to_delete or connection.output_node == node_to_delete:
                        connections_to_delete.append(connection)

                # Delete the node
                del self.nodes[node_to_delete]

                # Delete all the connections related to the node
                for connection in connections_to_delete:
                    del self.connections[connection.innovation_number]

                num_enabled_after = self.check_connection_enabled_amount()
                if num_enabled_after == 0:
                    raise Exception('You have removed all the connections due to a node removal')
                print('NODE BEING DELETED: ', node_to_delete)
                print('VIABLE NODES TO DELETE: ', viable_nodes_to_be_delete)
                return node_to_delete

    def compute_compatibility_distance(self, other_genome, config, generation_tracker=None):
        """
        Calculates the compabitility distance between two genomes
        :param config: Contains parameters
        :param other_genome: The other genome being compared with
        :param excess_coefficient:
        :param disjoint_coefficient:
        :param match_genes_coefficient:
        :return: The compatibility distance
        """
        max_num_genes = max(len(self.connections), len(other_genome.connections))
        # In the paper they state that if the max number of genes is less than 20 then you don't need to divide by a normalising value
        max_num_genes = max_num_genes if max_num_genes > 20 else 1

        num_excess = 0
        num_disjoint = 0
        matching_genes = []

        # This will be used to check if a gene not matching is disjoint or excess
        lowest_max_innovation_number = min(max(self.connections), max(other_genome.connections))

        # Contains a list of all innovation numbers for both genes
        unique_gene_innovation_numbers = set()

        for innovation_number in other_genome.connections:
            unique_gene_innovation_numbers.add(innovation_number)

        for innovation_number in self.connections:
            unique_gene_innovation_numbers.add(innovation_number)

        for innovation_number in unique_gene_innovation_numbers:
            genome_1_connection = self.connections.get(innovation_number)
            genome_2_connection = other_genome.connections.get(innovation_number)

            # If both genomes have it, it is a matching gene
            if genome_1_connection and genome_2_connection:
                matching_genes.append(abs(genome_1_connection.weight - genome_2_connection.weight))
            elif genome_1_connection or genome_2_connection:
                if innovation_number <= lowest_max_innovation_number:
                    num_disjoint += 1
                else:
                    num_excess += 1
            else:
                raise KeyError('This innovation number should have returned for one of the genomes')

        compatibility_distance = ((config.disjoint_coefficient * num_disjoint) / max_num_genes) + (
                (config.excess_coefficient * num_excess) / max_num_genes)

        if matching_genes:
            average_weight_diff = np.mean(matching_genes)
            compatibility_distance += (config.matching_genes_coefficient * average_weight_diff)

        if generation_tracker:
            generation_tracker.num_disjoint_list.append(num_disjoint)
            generation_tracker.num_excess_list.append(num_excess)
            if matching_genes:
                generation_tracker.weight_diff_list.append(average_weight_diff)

        return compatibility_distance

    def mutate_weight(self, config, generation_tracker, backprop_mutation=False):
        if not backprop_mutation:
            assert (config.weight_mutation_perturbe_chance + config.weight_mutation_reset_connection_chance == 1)
        else:
            assert (
                    config.weight_mutation_perturbe_chance_backprop + config.weight_mutation_reset_connection_chance_backprop == 1)

        if backprop_mutation:
            reset_all_connections_role = np.random.uniform(low=0.0, high=1.0)

        weight_mutation_mean = config.weight_mutation_mean_backprop if backprop_mutation else config.weight_mutation_mean
        weight_mutation_sigma = config.weight_mutation_sigma_backprop if backprop_mutation else config.weight_mutation_sigma

        # Mutate all connection weights
        for connection in self.connections.values():
            # If reset_all_connections isn't triggered, it'll just pertbe or reset the values as usual
            if backprop_mutation and reset_all_connections_role < config.weight_mutation_reset_all_connections_chance_backprop:
                # TODO: Delete this after debugging [16/04/19 @15:58]
                print('RESETING ALL THE WEIGHTS')
                connection.weight = np.random.randn()
            else:
                # This determines the chance for it to be a positive or negative change to the weight
                random_chance = np.random.uniform(low=0.0, high=1.0)
                assert (0 <= random_chance <= 1)
                perturbe_prob = config.weight_mutation_perturbe_chance_backprop if backprop_mutation else config.weight_mutation_perturbe_chance
                if random_chance < perturbe_prob:
                    perturbation_value = np.random.normal(loc=weight_mutation_mean,
                                                          scale=weight_mutation_sigma)
                    generation_tracker.perturbation_values_list.append(perturbation_value)
                    connection.weight += perturbation_value

                # 10% chance for the weight to be assigned a random weight
                else:
                    connection.weight = np.random.randn()

        for node in self.nodes.values():
            if node.node_type != 'source':
                # This determines the chance for it to be a positive or negative change to the weight
                random_chance = np.random.uniform(low=0.0, high=1.0)
                assert (0 <= random_chance <= 1)
                # 90% chance for the weight to be perturbed by a small amount
                if random_chance < 0.9:
                    perturbation_value = np.random.normal(loc=weight_mutation_mean,
                                                          scale=weight_mutation_sigma)
                    generation_tracker.perturbation_values_list.append(perturbation_value)
                    node.bias += perturbation_value

                # 10% chance for the weight to be assigned a random weight
                else:
                    node.bias = np.random.randn()

    def reset_all_connection_weights(self, config):
        # Mutate all connection weights
        for connection in self.connections.values():
            # Set all weights to a random value
            connection.weight = np.random.normal(loc=config.weight_mutation_mean,
                                                 scale=config.weight_mutation_sigma)


def main():
    node_list = [NodeGene(node_id=0, node_type='source'),
                 NodeGene(node_id=1, node_type='source'),
                 NodeGene(node_id=2, node_type='output', bias=1),
                 NodeGene(node_id=3, node_type='output', bias=1),
                 NodeGene(node_id=4, node_type='output', bias=1)]

    connection_list = [
        ConnectionGene(input_node=0, output_node=2, innovation_number=1, weight=-0.351, enabled=True),
        ConnectionGene(input_node=0, output_node=3, innovation_number=2, weight=-0.351, enabled=True),
        ConnectionGene(input_node=0, output_node=4, innovation_number=3, weight=-0.351, enabled=True),
        ConnectionGene(input_node=1, output_node=2, innovation_number=4, weight=-0.351, enabled=True),
        ConnectionGene(input_node=1, output_node=3, innovation_number=5, weight=-0.351, enabled=True),
        ConnectionGene(input_node=1, output_node=4, innovation_number=6, weight=-0.351, enabled=True)]

    genome = GenomeMultiClass(connections=connection_list, nodes=node_list, key=3)
    x_data, y_data = create_data(n_generated=500)
    genome_nn = GenomeNeuralNetwork(genome=genome, create_weights_bias_from_genome=False, activation_type='sigmoid',
                                    learning_rate=0.1,
                                    x_train=x_data, y_train=y_data)
    print(genome.num_layers_including_input)
    print(genome.constant_weight_connections)
    print(genome.layer_connections_dict)


if __name__ == "__main__":
    main()
