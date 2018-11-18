from deconstruct_genome import DeconstructGenome
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
        assert (len(self.layer_nodes[self.num_layers_including_input]) == 1)

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
        assert (genome_1.fitness != genome_2.fitness)

        if genome_1.fitness > genome_2.fitness:
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

    def mutate(self, new_innovation_number):
        """
        Will call one of the possible mutation abilities using a random() number generated
        :return:
        """
        # The innovation is unique so it should not already exist in the list of connections
        assert (new_innovation_number not in self.connections)

        # Unpack the genome after whatever mutation has occured
        self.unpack_genome()
        pass

    def get_new_connection_gene(self, new_innovation_number):
        viable_start_nodes = []

        # Any node that isn't the output node is a viable start_node
        for node_id, node in self.nodes.items():
            if node.node_type != 'output':
                viable_start_nodes.append(node_id)

        # Pick a random node for the start node
        start_node = random.choice(viable_start_nodes)

        suitable_end_nodes = []

        start_node_layer = self.node_layers[start_node]

        # Any node on the current layer for the start node or any node after that layer is a suitable end node as
        # long as it's not a source node
        if self.nodes[start_node].node_type != 'source':
            for layer in range(start_node_layer, self.num_layers_including_input + 1):
                suitable_end_nodes += self.layer_nodes[layer]
        # If it's a source node the layer has to be any node in a layer after itself TODO: IS this correct?
        else:
            for layer in range(start_node_layer + 1, self.num_layers_including_input + 1):
                suitable_end_nodes += self.layer_nodes[layer]

        # Can't connect to itself
        if start_node in set(suitable_end_nodes):
            suitable_end_nodes.remove(start_node)

        # Pick a random node for the end node
        end_node = random.choice(suitable_end_nodes)

        # Create the new connection gene
        new_connection_gene = ConnectionGene(input_node=start_node, output_node=end_node, weight=np.random.randn(),
                                             innovation_number=new_innovation_number)

        return new_connection_gene

    def add_connection(self, new_innovation_number):
        """
        Add a random connection
        :param new_innovation_number: The innovation number to be assigned to the new connection gene
        """

        # If there are no hidden nodes then there's no unique new connection we can add that doesn't exist
        num_hidden_nodes = 0
        for node in self.nodes.values():
            if node.node_type == 'hidden':
                num_hidden_nodes += 1
        # TODO: Becareful with this because if there are no hidden nodes then this won't ever break.
        # We need to check if the connection already exists as we don't want to re-create a connection
        if num_hidden_nodes > 0:
            while True:
                connection_already_exists = False
                new_connection_gene = self.get_new_connection_gene(new_innovation_number=new_innovation_number)
                for connection in self.connections.values():
                    if connection.input_node == new_connection_gene.input_node and connection.output_node == new_connection_gene.output_node:
                        connection_already_exists = True
                # Restart the loop if exists
                if connection_already_exists:
                    continue
                # Else break the while loop
                else:
                    break

            # Add the connection the the genome
            self.connections[new_connection_gene.innovation_number] = new_connection_gene

            return new_connection_gene

    def remove_connection(self):
        """
        Remove a random existing connection from the genome
        """
        viable_connection_to_remove = []

        source_input_node_connections = dict()
        # TODO: NEAT doesn't have remove node or connection

        for connection in self.connections.values():
            if self.nodes[connection.input_node].node_type == 'source':
                if connection.input_node not in source_input_node_connections:
                    source_input_node_connections[connection.input_node] = 1
                else:
                    source_input_node_connections[connection.input_node] += 1

        for connection in self.connections.values():

            # Any connection that doesn't include the source node can be removed # TODO: Is this correct?
            if self.nodes[connection.input_node].node_type != 'source':
                viable_connection_to_remove.append(connection.innovation_number)

        # Pick a random connection to remove
        connection_to_remove = random.choice(viable_connection_to_remove)

        # Delete the connection
        del self.connections[connection_to_remove]

    def add_node(self, new_innovation_number):
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

        # Create new connection genes
        first_new_connection = ConnectionGene(input_node=input_node, output_node=new_node.node_id,
                                              innovation_number=new_innovation_number)
        second_new_connection = ConnectionGene(input_node=new_node.node_id, output_node=output_node,
                                               innovation_number=new_innovation_number + 1)

        self.connections[first_new_connection.innovation_number] = first_new_connection
        self.connections[second_new_connection.innovation_number] = second_new_connection

        return first_new_connection, second_new_connection

    def delete_node(self):
        viable_nodes_to_be_delete = []

        for node in self.nodes.values():
            # Any node that isn't the source or output node can be deleted
            if node.node_type != 'source' or node.node_type != 'output':
                viable_nodes_to_be_delete.append(node.node_id)

        # Randomly choose node to delete
        node_to_delete = random.choice(viable_nodes_to_be_delete)

        # Delete the node
        del self.nodes[node_to_delete]

    def compute_compatibility_distance(self, other_genome, excess_coefficient, disjoint_coefficient,
                                       match_genes_coefficient):
        """
        Calculates the compabitility distance between two genomes
        :param other_genome: The other genome being compared with
        :param excess_coefficient:
        :param disjoint_coefficient:
        :param match_genes_coefficient:
        :return: The compatibility distance
        """
        max_num_genes = max(len(self.connections), len(other_genome.connections))

        num_excess = 0
        num_disjoint = 0
        matching_genes = list()

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
                if innovation_number < lowest_max_innovation_number:
                    num_disjoint += 1
                else:
                    num_excess += 1
            else:
                raise KeyError('This innovation number should have returned for one of the connections')

        average_weight_diff = np.mean(matching_genes)

        compatibility_distance = ((disjoint_coefficient * num_disjoint) / max_num_genes) + (
                (excess_coefficient * num_excess) / max_num_genes) + (match_genes_coefficient * average_weight_diff)

        return compatibility_distance


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
