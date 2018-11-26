from deconstruct_genome import DeconstructGenome
from graph_algorithm import Graph
import itertools
import numpy as np
from gene import ConnectionGene, NodeGene
import random


class Genome:
    def __init__(self, key, connections=None, nodes=None):
        """
        :param key: Which genome number it is
        :param connections: A list of ConnectionGene instances
        :param nodes: A list of NodeGene Instance
        """

        # Unique identifier for a genome instance.
        self.key = key

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

    def unpack_genome(self):
        """
        Deconstructs the genome into a structure that can be used for the neural network it represents
        """
        # Unpack the genome and get the returning dictionary
        return_dict = DeconstructGenome.unpack_genome(genome=self)

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

        # The last layer should only contain the output node
        if len(self.layer_nodes[self.num_layers_including_input]) != 1:
            raise Exception('Invalid genome has been unpacked')


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

    def crossover(self, genome_1, genome_2):
        """
        :param genome_1:
        :param genome_2:
        :return:
        """
        assert isinstance(genome_1.fitness, (int, float))
        assert isinstance(genome_2.fitness, (int, float))
        # They should never be EXACTLY the same. But if this ever gets triggered you should write the crossover case for
        # it

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
                inherited_gene = random.choice(connection_genes)
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

        # Unpack the genome after we have configured it
        self.unpack_genome()

    def mutate(self, new_innovation_number, current_gen_innovations, config):
        """
        Will call one of the possible mutation abilities using a random() number generated
        :return:
        """
        # The innovation is unique so it should not already exist in the list of connections
        assert (new_innovation_number not in self.connections)

        # The rolls to see if each mutation occurs
        add_connection_roll = np.random.uniform(low=0.0, high=1.0)
        add_node_roll = np.random.uniform(low=0.0, high=1.0)
        mutate_weight_roll = np.random.uniform(low=0.0, high=1.0)
        remove_node_roll = np.random.uniform(low=0.0, high=1.0)
        remove_connection_roll = np.random.uniform(low=0.0, high=1.0)

        # Add connection if
        if add_connection_roll < config.add_connection_mutation_chance:
            self.add_connection(new_innovation_number=new_innovation_number,
                                current_gen_innovations=current_gen_innovations)
        # Add node if
        if add_node_roll < config.add_node_mutation_chance:
            self.add_node(new_innovation_number=new_innovation_number, current_gen_innovations=current_gen_innovations)
        # Mutate weight if
        if mutate_weight_roll < config.weight_mutation_chance:
            self.mutate_weight(config=config)

        if remove_node_roll < config.remove_node_mutation_chance:
            self.remove_node()

        if remove_connection_roll < config.remove_connection_mutation_chance:
            self.remove_connection()

        # Unpack the genome after whatever mutation has occured
        self.unpack_genome()

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

            input_node_type = self.nodes[input_node].node_type

            input_node_layer = self.node_layers[input_node]
            output_node_layer = self.node_layers[output_node]

            # As long as the node type isn't a source then the connecting node can be on the same layer or greater
            if input_node_type != 'source' and output_node_layer >= input_node_layer:
                cleaned_combinations.append(combination)
            # If the input node type is a source then the connecting node can't be on the same layer
            elif input_node_type == 'source' and output_node_layer > input_node_layer:
                cleaned_combinations.append(combination)

        return cleaned_combinations

    def add_connection(self, new_innovation_number, current_gen_innovations):
        """
        Add a random connection
        :param new_innovation_number: The innovation number to be assigned to the new connection gene
        """

        # Keeps a list of nodes which can't be chosen from to be the start node of the new connection
        viable_start_nodes = []

        # Any node that isn't the output node is a viable start_node
        for node_id, node in self.nodes.items():
            if node.node_type != 'output':
                viable_start_nodes.append(node_id)

        possible_combinations = itertools.combinations(list(self.nodes.keys()), 2)
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

            if new_connection in current_gen_innovations:
                # If the connection was already made then use whatever the innovation number that was assigned to that connection
                new_connection_gene = ConnectionGene(input_node=new_connection[0], output_node=new_connection[1],
                                                     innovation_number=current_gen_innovations[new_connection],
                                                     weight=np.random.random())

            else:
                new_connection_gene = ConnectionGene(input_node=new_connection[0], output_node=new_connection[1],
                                                     innovation_number=new_innovation_number, weight=np.random.random())

                # Save the connection as a current gen innovation
                current_gen_innovations[new_connection] = new_innovation_number

            # Add the connection the the genome
            self.connections[new_connection_gene.innovation_number] = new_connection_gene

            return new_connection_gene
        else:
            print('no new connection possible')

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
        # Check to see that there is at least two source connections and two output connections so that if one get's
        # removed it's still a valid neural network
        num_source_connections = 0
        num_output_connections = 0
        # All the connections that aren't linked to the output
        connections_excluding_output = []
        # All the connections that aren't linked to the source
        connections_excluding_source = []
        # All the connections that aren't linked to either source or output
        connections_excluding_source_and_output = []
        for connection in self.connections.values():
            # TODO: Check for other mutations about connection being enabled
            # Add all the connections which don't have the output as the output node
            if self.nodes[connection.output_node].node_type == 'output':
                # Only counts if the connection is enabled
                if connection.enabled:
                    num_output_connections += 1
            else:
                connections_excluding_output.append(connection)

            if self.nodes[connection.input_node].node_type == 'source':
                # Only counts if the connection is enabled
                if connection.enabled:
                    num_source_connections += 1
            else:
                connections_excluding_source.append(connection)

            if self.nodes[connection.input_node] != 'source' and self.nodes[connection.output_node] != 'output':
                connections_excluding_source_and_output.append(connection)

        # Means it doesn't matter which one is picked, there will still be a valid path to the end, so we can choose any
        if num_source_connections > 1 and num_output_connections > 1:
            choice_list = list(self.connections.values())
        # Means that there aren't enough source connections so we can't remove any of them
        elif num_output_connections > 1:
            choice_list = connections_excluding_source
        # Means there aren't enough output connections so we can't remove any
        elif num_source_connections > 1:
            choice_list = connections_excluding_output
        # Means there aren't enough source and output connections so we can't choose any of those connections to remove
        else:
            choice_list = connections_excluding_source_and_output

        return choice_list

    def check_num_paths(self):
        graph = Graph()
        source_nodes = []
        output_nodes = []
        # Add the connections to the graph
        for connection in self.connections.values():
            graph.add_edge(start_node=connection.input_node, end_node=connection.output_node)
            input_node = self.nodes[connection.input_node]
            output_node = self.nodes[connection.output_node]
            if input_node.node_type == 'source':
                source_nodes.append(input_node)
            if output_node.node_type == 'output':
                output_nodes.append(output_node)

        # Only keep the unique nodes to there are no duplicates in the list
        source_nodes = list(set(source_nodes))
        output_nodes = list(set(output_nodes))

        # There shouldn't be more than one output node
        assert (len(output_nodes) == 1)

        # Keeps track of how many paths there are from the source to the input's
        num_source_to_output_paths = 0
        for node in source_nodes:
            num_source_to_output_paths += graph.count_paths(start_node=node.node_id, end_node=output_nodes[0].node_id)

        return num_source_to_output_paths

    def remove_connection(self):
        """
        Removes a random existing connection form the genome
        """

        num_source_to_output_paths = self.check_num_paths()

        # If there is only one path from the source to the output we shouldn't delete any of the connections since
        # it would make it an invalid network
        if num_source_to_output_paths > 1:
            choice_list = self.check_which_connections_removable()
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

    def add_node(self, new_innovation_number, current_gen_innovations):
        """
        Add a node between two existing nodes
        """

        # Create the new node
        new_node = NodeGene(node_id=(max(self.nodes.keys()) + 1), node_type='hidden', bias=np.random.randn())

        # The connection where the node will be added
        connection_to_add_node = random.choice(list(self.connections.values()))
        # Disable the connection since it will be replaced
        connection_to_add_node.enabled = False

        input_node = connection_to_add_node.input_node
        output_node = connection_to_add_node.output_node

        # Create new connection gene. Which has a weight of 1
        first_combination = (input_node, new_node.node_id)
        if first_combination in current_gen_innovations:
            first_new_connection = ConnectionGene(input_node=input_node, output_node=new_node.node_id,
                                                  innovation_number=current_gen_innovations[first_combination],
                                                  weight=1)
        else:
            first_new_connection = ConnectionGene(input_node=input_node, output_node=new_node.node_id,
                                                  innovation_number=new_innovation_number,
                                                  weight=1)
            # Save the innovation since it's new
            current_gen_innovations[first_combination] = new_innovation_number
        second_combination = (new_node.node_id, output_node)

        # The second connection keeps the weight of the connection it replaced
        if second_combination in current_gen_innovations:
            second_new_connection = ConnectionGene(input_node=new_node.node_id, output_node=output_node,
                                                   innovation_number=current_gen_innovations[second_combination],
                                                   weight=connection_to_add_node.weight)
        else:
            second_new_connection = ConnectionGene(input_node=new_node.node_id, output_node=output_node,
                                                   innovation_number=new_innovation_number + 1,
                                                   weight=connection_to_add_node.weight)
            # save the innovation if it doesn't already exist
            current_gen_innovations[second_combination] = new_innovation_number + 1

        # Add the new node and connections
        self.nodes[new_node.node_id] = new_node
        self.connections[first_new_connection.innovation_number] = first_new_connection
        self.connections[second_new_connection.innovation_number] = second_new_connection

        return first_new_connection, second_new_connection

    def remove_node(self):
        num_source_to_output_paths = self.check_num_paths()

        # Otherwise if you delete a node and it only has one path, you make it an invalid network as there is not path
        # from the source to the output
        if num_source_to_output_paths > 1:
            viable_nodes_to_be_delete = []
            connections_able_to_remove = self.check_which_connections_removable()

            # for node in self.nodes.values():
            #     # Any node that isn't the source or output node can be deleted
            #     if node.node_type != 'source' or node.node_type != 'output':
            #         viable_nodes_to_be_delete.append(node.node_id)

            for connection in connections_able_to_remove:
                if self.nodes[connection.input_node].node_type != 'source':
                    viable_nodes_to_be_delete.append(connection.input_node)
                if self.nodes[connection.output_node].node_type != 'output':
                    viable_nodes_to_be_delete.append(connection.output_node)

            # Remove duplicates
            viable_nodes_to_be_delete = list(set(viable_nodes_to_be_delete))

            # If there are any viable nodes to delete
            if viable_nodes_to_be_delete:

                # Randomly choose node to delete
                node_to_delete = random.choice(viable_nodes_to_be_delete)

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

    def compute_compatibility_distance(self, other_genome, config):
        """
        Calculates the compabitility distance between two genomes
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

            if genome_1_connection and genome_2_connection:
                matching_genes.append(abs(genome_1_connection.weight - genome_2_connection.weight))
            elif genome_1_connection or genome_2_connection:
                if innovation_number <= lowest_max_innovation_number:
                    num_disjoint += 1
                else:
                    num_excess += 1
            else:
                raise KeyError('This innovation number should have returned for one of the genomes')

        average_weight_diff = np.mean(matching_genes)

        compatibility_distance = ((config.disjoint_coefficient * num_disjoint) / max_num_genes) + (
                (config.excess_coefficient * num_excess) / max_num_genes) + (
                                         config.matching_genes_coefficient * average_weight_diff)

        return compatibility_distance

    def mutate_weight(self, config):

        # Mutate all connection weights
        for connection in self.connections.values():
            random_chance = np.random.uniform(low=0.0, high=1.0)
            assert (0 <= random_chance <= 1)
            # 90% chance for the weight to be perturbed by a small amount
            if random_chance < 0.9:
                # Choose randomly from a range to change the value by
                connection.weight += np.random.uniform(low=config.weight_range_low, high=config.weight_range_high)
            # 10% chance for the weight to be assigned a random weight
            else:
                connection.weight = np.random.randn()


def main():
    node_list = [NodeGene(node_id=1, node_type='source'),
                 NodeGene(node_id=2, node_type='source'),
                 NodeGene(node_id=3, node_type='hidden'),
                 NodeGene(node_id=4, node_type='hidden'),
                 NodeGene(node_id=5, node_type='output')]

    # Note that one of the connections isn't enabled
    connection_list = [ConnectionGene(input_node=1, output_node=5, innovation_number=1, enabled=True),
                       ConnectionGene(input_node=1, output_node=4, innovation_number=2, enabled=True),
                       ConnectionGene(input_node=2, output_node=3, innovation_number=3, enabled=True),
                       ConnectionGene(input_node=2, output_node=4, innovation_number=4, enabled=True),
                       ConnectionGene(input_node=3, output_node=4, innovation_number=7, enabled=True),
                       ConnectionGene(input_node=3, output_node=5, innovation_number=8, enabled=True),
                       ConnectionGene(input_node=4, output_node=5, innovation_number=6, enabled=True)]

    genome = Genome(connections=connection_list, nodes=node_list, key=3)

    print(genome.num_layers_including_input)
    print(genome.constant_weight_connections)
    print(genome.layer_connections_dict)


if __name__ == "__main__":
    main()
