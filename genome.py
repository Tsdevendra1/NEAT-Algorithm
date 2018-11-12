
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

    def configure_genes(self, connections, nodes):
        # (innovation_number, ConnectionGene class object) pairs for connection gene sets.
        for connection in connections:
            if connection.innovation_number in self.connections:
                raise KeyError('You do not have a unique innovation number for this connection')
            self.connections[connection.innovation_number] = connection

        # (node_id, NodeGene class object) pairs for the node gene set
        for node in nodes:
            self.nodes[node.node_id] = node




