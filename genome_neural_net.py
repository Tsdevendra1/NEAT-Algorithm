from neural_network import NeuralNetwork
import collections
from gene import *
from genome import *


class DeconstructGenome:

    @classmethod
    def unpack_genome(cls, genome):
        nodes = genome.nodes.values()
        connections = genome.connections.values()
        # Get's which node is on which layer
        node_layers, layer_nodes = cls.get_node_layers(connections=connections, num_nodes=len(nodes))
        num_layers = max(node_layers.values())
        nodes_per_layer = collections.Counter(list(node_layers.values()))
        node_map = cls.get_node_map(num_layers=num_layers, layer_nodes=layer_nodes)
        added_nodes, added_connections = cls.find_ghost_nodes(nodes_per_layer=nodes_per_layer, node_layers=node_layers,
                                                              node_map=node_map, connections=connections, nodes=nodes)
        return node_map

    @classmethod
    def get_node_layers(cls, connections, num_nodes):
        """
        :param num_nodes: number of nodes in the genome
        :param connections: list of connections from the genome
        :return: A dictinary showing which layer number each node is on
        """
        # Keeps track of which node is in which layer. (Have to add one because of python indexing starting at 0)
        node_layers = {key: 0 for key in list(range(1, num_nodes + 1))}
        # Will be used to keep track of progress of finding which node is in which layer
        old_layers = {key: -1 for key in list(range(1, num_nodes + 1))}

        # Until there is no change between our last guess and the next one
        while old_layers != node_layers:
            old_layers = node_layers
            for connection in connections:
                # Update the index of the input node to the max between 1 or the current value at the index
                node_layers[connection.input_node] = max(1, node_layers[connection.input_node])
                # Update the index of the output node between what the input layer is +1 or the current value of the
                # output node
                node_layers[connection.output_node] = max(node_layers[connection.output_node],
                                                          (node_layers[connection.input_node] + 1))

        layer_nodes = {key: [] for key in node_layers.values()}
        for node_id, layer in node_layers.items():
            layer_nodes[layer].append(node_id)

        return node_layers, layer_nodes

    @classmethod
    def get_node_map(cls, num_layers, layer_nodes):
        node_map = dict()
        for layer in range(1, num_layers + 1):
            counter = 1
            # We go through each index of a node for each layer at assign it an incremented number.
            for node_id in layer_nodes[layer]:
                node_map[node_id] = counter
                counter += 1
        return node_map

    @classmethod
    def find_active_nodes(cls, genome):
        pass

    @classmethod
    def find_ghost_nodes(cls, nodes_per_layer, node_layers, node_map, connections, nodes):
        added_nodes = []
        added_connections = []

        for connection in connections:
            # How many layers the connection spans
            input_node_layer = node_layers[connection.input_node]
            output_node_layer = node_layers[connection.output_node]
            layer_span = output_node_layer - input_node_layer

            if layer_span > 1:
                num_added_nodes = layer_span - 1
                # # Get the highest value node_id because we will use this as the starting point for the next nodes
                # last_node_id = max(node_layers.keys())
                # # A list of what the node id's will be for the newly added nodes
                # new_node_ids = list(range(last_node_id + 1, last_node_id + num_added_nodes))
                # # What layers the new nodes will be in
                # layers_for_new_nodes = list(range(input_node_layer + 1, input_node_layer + num_added_nodes + 1))
                #
                # # Saves which number node they are in the layer
                # node_numbers = []
                # for layer in layers_for_new_nodes:
                #     node_numbers.append((nodes_per_layer[layer] + 1))
                #
                # # They should be the same length since they go hand in hand
                # assert (len(layers_for_new_nodes) == len(new_node_ids))
                # assert (len(layers_for_new_nodes) == len(node_numbers))
                #
                # for index in range(num_added_nodes):
                #     node_id = new_node_ids[index]
                #     layer = layers_for_new_nodes[index]
                #     node_number_in_layer = node_numbers[index]
                #
                #     # Saving which layer the node_id is on
                #     node_layers[node_id] = layer
                #     # Updating the number of nodes for the layer
                #     nodes_per_layer[layer] += 1
                #     # Saving which node it is in it's respective layer
                #     node_map[node_id] = node_number_in_layer
                new_nodes = cls.update_nodes(num_added_nodes=num_added_nodes, added_nodes=added_nodes, nodes=nodes,
                                             node_layers=node_layers, input_node_layer=input_node_layer,
                                             nodes_per_layer=nodes_per_layer, node_map=node_map)
                # TODO: Update the active nodes and biases?

                # Turn off the connection we're in
                connection.enabled = False
                cls.update_connections(new_node_ids=new_nodes, connection_gene=connection, connections=connections,
                                       added_connections=added_connections)
                # new_input_nodes = [connection.input_node] + new_nodes_ids
                # new_output_nodes = new_node_ids + [connection.output_node]
                # num_new_connections = len(new_input_nodes)
                # Add the new nodes
                # for new_node in range(num_added_nodes):
                #     new_node_gene = NodeGene(node_id=new_node_ids[new_node], node_type='hidden')
                #     # TODO: Setting weights for the connection. And setting innovation numbers for the genes? What purpose does the connection gene hold because surely we only car eabout innovation number
                #     nodes.append(new_node_gene)
                #     # Keep track of the new nodes we've added
                #     added_nodes.append(new_node_gene)

                # # Add the new connections
                # for new_connection in range(num_new_connections):
                #     new_connection_gene = ConnectionGene(input_node=new_input_nodes[new_connection],
                #                                          output_node=new_output_nodes[new_connection])
                #     connections.append(new_connection_gene)
                #     # Keep track of the new connections we've added
                #     added_connections.append(new_connection_gene)

        return added_nodes, added_connections

    @classmethod
    def update_nodes(cls, num_added_nodes, added_nodes, nodes, node_layers, input_node_layer, nodes_per_layer,
                     node_map):
        # Get the highest value node_id because we will use this as the starting point for the next nodes
        last_node_id = max(node_layers.keys())
        # A list of what the node id's will be for the newly added nodes
        new_node_ids = list(range(last_node_id + 1, last_node_id + num_added_nodes))
        # What layers the new nodes will be in
        layers_for_new_nodes = list(range(input_node_layer + 1, input_node_layer + num_added_nodes + 1))

        # Saves which number node they are in the layer
        node_numbers = []
        for layer in layers_for_new_nodes:
            node_numbers.append((nodes_per_layer[layer] + 1))

        # They should be the same length since they go hand in hand
        assert (len(layers_for_new_nodes) == len(new_node_ids))
        assert (len(layers_for_new_nodes) == len(node_numbers))

        for index in range(num_added_nodes):
            node_id = new_node_ids[index]
            layer = layers_for_new_nodes[index]
            node_number_in_layer = node_numbers[index]

            # Saving which layer the node_id is on
            node_layers[node_id] = layer
            # Updating the number of nodes for the layer
            nodes_per_layer[layer] += 1
            # Saving which node it is in it's respective layer
            node_map[node_id] = node_number_in_layer

            new_node_gene = NodeGene(node_id=node_id, node_type='hidden')
            # TODO: Setting weights for the connection. And setting innovation numbers for the genes? What purpose does the connection gene hold because surely we only car eabout innovation number
            nodes.append(new_node_gene)
            # Keep track of the new nodes we've added
            added_nodes.append(new_node_gene)

        return new_node_ids

    @classmethod
    def update_connections(cls, new_node_ids, connection_gene, added_connections, connections):
        connection_gene.enabled = False
        new_input_nodes = [connection_gene.input_node] + new_node_ids
        new_output_nodes = new_node_ids + [connection_gene.output_node]
        num_new_connections = len(new_input_nodes)
        for new_connection in range(num_new_connections):
            new_connection_gene = ConnectionGene(input_node=new_input_nodes[new_connection],
                                                 output_node=new_output_nodes[new_connection])
            connections.append(new_connection_gene)
            # Keep track of the new connections we've added
            added_connections.append(new_connection_gene)


class GenomeNeuralNet(NeuralNetwork):

    def __init__(self, ):
        pass


def main():
    node_list = [NodeGene(node_id=1, node_type='source'),
                 NodeGene(node_id=2, node_type='source'),
                 NodeGene(node_id=3, node_type='hidden'),
                 NodeGene(node_id=4, node_type='hidden'),
                 NodeGene(node_id=5, node_type='output')]

    connection_list = [ConnectionGene(input_node=1, output_node=3, innovation_number=1, enabled=True),
                       ConnectionGene(input_node=1, output_node=4, innovation_number=2, enabled=True),
                       ConnectionGene(input_node=2, output_node=3, innovation_number=3, enabled=True),
                       ConnectionGene(input_node=2, output_node=4, innovation_number=4, enabled=True),
                       ConnectionGene(input_node=3, output_node=5, innovation_number=5, enabled=True),
                       ConnectionGene(input_node=4, output_node=5, innovation_number=6, enabled=True)]

    genome = Genome(nodes=node_list, connections=connection_list, key=1)

    print(DeconstructGenome.unpack_genome(genome))


if __name__ == "__main__":
    main()
