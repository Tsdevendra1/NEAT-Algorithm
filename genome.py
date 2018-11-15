from deconstruct_genome import DeconstructGenome
import numpy as np
from gene import ConnectionGene, NodeGene
import random


class Genome:
    def __init__(self, key, connections=None, nodes=None, unpack_genome=True):
        """
        :param key: Which genome number it is
        :param connections: A list of ConnectionGene instances
        :param nodes: A list of NodeGene Instance
        """
        assert isinstance(connections, list)
        assert isinstance(nodes, list)

        # Unique identifier for a genome instance.
        self.key = key

        self.connections = {}
        self.nodes = {}

        # Saves the genes in an appropriate format into the dictionaries above. (See the method for saved format)
        self.configure_genes(connections=connections, nodes=nodes)

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
        self.num_layers_including_input = None

        if unpack_genome:
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
        self.layer_nodes = return_dict['layer_nodes']
        self.num_layers_including_input = max(self.layer_nodes)

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
        pass

    def mutate(self, new_innovation_number):
        """
        Will call one of the possible mutation abilities using a random() number generated
        :return:
        """
        pass

    def add_connection(self, new_innovation_number):
        """
        Add a random connection
        :param new_innovation_number: The innovation number to be assigned to the new connection gene
        """
        viable_start_nodes = []

        # Any node that isn't the source node is a viable start_node
        for node_id, node in self.nodes.items():
            if node.node_type != 'source':
                viable_start_nodes.append(node_id)

        # Pick a random node for the start node
        start_node = random.choice(viable_start_nodes)

        suitable_end_nodes = []

        start_node_layer = self.node_map[start_node]

        # Any node on the current layer for the start node or any node after that layer is a suitable end node
        for layer in range(start_node_layer, self.num_layers_including_input + 1):
            suitable_end_nodes += self.layer_nodes[layer]

        # Pick a random node for the end node
        end_node = random.choice(suitable_end_nodes)

        # Create the new connection gene
        new_connection_gene = ConnectionGene(input_node=start_node, output_node=end_node, weight=np.random.randn(),
                                             innovation_number=new_innovation_number)

        # Add the connection the the genome
        self.connections[new_connection_gene.innovation_number] = new_connection_gene

    def remove_connection(self):
        """
        Remove a random existing connection from the genome
        """
        # Pick a random connection to remove
        connection_to_remove = random.choice(list(self.connections.values()))

        # Delete the connection
        del self.connections[connection_to_remove.innovation_number]

    def add_node(self, new_innovation_number):
        """
        Add a node between two existing nodes
        """

        new_node = NodeGene(node_id=max(self.nodes.keys()), node_type='hidden', bias=np.random.randn())

        # The connection where the node will be added
        connection_to_add_node = random.choice(self.connections.values())
        input_node = connection_to_add_node.input_node
        output_node = connection_to_add_node.output_node

        # Create new connection genes
        first_new_connection = ConnectionGene(input_node=input_node, output_node=new_node.node_id,
                                              innovation_number=new_innovation_number)
        second_new_connection = ConnectionGene(input_node=new_node.node_id, output_node=output_node,
                                               innovation_number=new_innovation_number + 1)

        self.connections[first_new_connection.innovation_number] = first_new_connection
        self.connections[second_new_connection.innovation_number] = second_new_connection
