from deconstruct_genome import DeconstructGenome


class Genome:
    def __init__(self, key, connections, nodes):
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

        # Unpack genome for later use in neural network and mutations/crossovers
        self.connection_matrices_per_layer, self.no_activations_matrix_per_layer, self.constant_weight_connections, \
        self.nodes_per_layer, self.node_map, self.layer_connections_dict, self.updated_nodes, self.layer_nodes \
            = DeconstructGenome.unpack_genome(genome=self)

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
