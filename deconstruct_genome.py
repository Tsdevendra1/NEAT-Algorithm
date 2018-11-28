from neural_network import NeuralNetwork
import copy
import collections
from gene import *
import numpy as np


class DeconstructGenome:

    @classmethod
    def unpack_genome(cls, genome, all_paths):
        """
        :param genome: The genome to be deconstructed
        :return: See GenomeNeuralNetwork class for details
        """
        # We use deep copies because we don't want to make changes to the genome itself
        nodes = copy.deepcopy(genome.nodes)
        connections = copy.deepcopy(list(genome.connections.values()))
        connections = cls.sort_connections(connections, nodes=nodes, all_paths=all_paths)

        # Get's which node is on which layer
        node_layers, layer_nodes = cls.get_node_layers(connections=connections, nodes=nodes)
        num_layers = max(node_layers.values())
        # Keeps track of number of nodes for each layer
        nodes_per_layer = dict(collections.Counter(list(node_layers.values())))
        # Keeps track of which number node each node is in their respective layer
        node_map, source_nodes = cls.get_node_map(num_layers=num_layers, layer_nodes=layer_nodes, nodes=nodes)
        added_nodes, added_connections, last_dummy_related_to_connection = cls.find_ghost_nodes(
            nodes_per_layer=nodes_per_layer, node_layers=node_layers,
            node_map=node_map, connections=connections,
            layer_nodes=layer_nodes)
        # Update the new connections and nodes
        for node in added_nodes:
            nodes[node.node_id] = node
        connections += added_connections

        connection_matrices, bias_matrices, constant_weight_connections, layer_connections_dict = cls.all_connection_matrices(
            connections=connections, layer_nodes=layer_nodes,
            node_layers=node_layers, num_layers=num_layers,
            node_map=node_map)

        output_node = None
        for node in nodes.values():
            if node.node_type == 'output':
                output_node = node
                break
        cls.confirm_correct_configuration(output_node=output_node.node_id, source_nodes=source_nodes, layer_nodes=layer_nodes)

        # Saves all the variables being returned from the function
        return_dict = {}
        return_dict['connection_matrices'] = connection_matrices
        return_dict['bias_matrices'] = bias_matrices
        return_dict['constant_weight_connections'] = constant_weight_connections
        return_dict['nodes_per_layer'] = nodes_per_layer
        return_dict['node_map'] = node_map
        return_dict['layer_connections_dict'] = layer_connections_dict
        return_dict['nodes'] = nodes
        return_dict['layer_nodes'] = layer_nodes
        return_dict['node_layers'] = node_layers
        return_dict['last_dummy_related_to_connection'] = last_dummy_related_to_connection
        return return_dict

    @classmethod
    def confirm_correct_configuration(cls, output_node, source_nodes, layer_nodes):
        """
        The source nodes should always be in the first layer and the output node should always be in the last layer
        :param output_node: The output node
        :param source_nodes: A list of the source_nodes
        :param layer_nodes: A dict containing which nodes are in which layer
        """
        for node in source_nodes:
            assert (node in layer_nodes[1])

        assert (output_node in layer_nodes[max(layer_nodes)])

    @classmethod
    def sort_connections(cls, connections, nodes, all_paths):
        """
        In order for the unpack genome method to work the connections list must be sorted such that the connections
        which connect to the output layer must always be last in the list
        :param all_paths: A list of all the paths from the source nodes to the output nodes
        :param nodes: A dictionary of the nodes (node_id, node_gene)
        :param connections: The list of connection genes
        :return: Sorted connections list
        """

        # sorted_list = []
        added_to_list = set()

        connections_by_tuple = {}
        for connection in connections:
            connections_by_tuple[(connection.input_node, connection.output_node)] = connection

        middle_node_connections = []
        # For all the paths for a given source node
        for source_node_paths in all_paths:
            # For each path for a given source node
            for path in source_node_paths:
                # Go through the connections for the path
                for index in range(len(path) - 1):
                    input_node = path[index]
                    output_node = path[index + 1]

                    # The connections which have a source or output node will be collected later
                    if nodes[input_node].node_type != 'source' and nodes[output_node].node_type != 'output':
                        connection_tuple = (input_node, output_node)

                        if connection_tuple not in added_to_list and connections_by_tuple[connection_tuple].enabled:
                            middle_node_connections.append(connections_by_tuple[connection_tuple])
                            added_to_list.add(connection_tuple)

        source_node_connections = []

        # First we add the input connections
        for connection in connections:
            # Check the connection is enabled and that we haven't already added it to the list
            if connection.enabled and connection not in added_to_list:
                input_node_type = nodes[connection.input_node].node_type
                if input_node_type == 'source':
                    source_node_connections.append(connection)
                    added_to_list.add(connection)

        # #
        # for connection in connections:
        #     # check the connection is enabled and that we haven't already added it to the list
        #     if connection.enabled and connection not in added_to_list:
        #         input_node_type = nodes[connection.input_node].node_type
        #         output_node_type = nodes[connection.output_node].node_type
        #         if output_node_type != 'output' and input_node_type != 'source':
        #             sorted_list.append(connection)
        #             added_to_list.add(connection)

        output_node_connections = []

        # Then we add the ones left (the ones linked to the output)
        for connection in connections:
            # check the connection is enabled and that we haven't already added it to the list
            if connection.enabled and connection not in added_to_list:
                output_node_type = nodes[connection.output_node].node_type
                if output_node_type == 'output':
                    output_node_connections.append(connection)
                    added_to_list.add(connection)

        sorted_list = source_node_connections + middle_node_connections + output_node_connections

        if not sorted_list:
            raise Exception('There are no connections specified for this genome')

        return sorted_list

    @classmethod
    def get_node_layers(cls, connections, nodes):
        """
        :param nodes : dict containing (node_id, node_gene)
        :param connections: list of connections from the genome
        :return: node_layers: A dictionary containing which layer each node is on, layer_nodes: A dictionary containing
                a list for each layer of what nodes are in that layer
        """

        active_nodes = set()

        for connection in connections:
            if connection.enabled:
                if connection.input_node not in active_nodes:
                    active_nodes.add(connection.input_node)
                if connection.output_node not in active_nodes:
                    active_nodes.add(connection.output_node)

        # Keeps track of which node is in which layer. (Have to add one because of python indexing starting at 0)
        node_layers = {key: 0 for key in list(active_nodes)}
        # Will be used to keep track of progress of finding which node is in which layer
        old_layers = {key: -1 for key in list(active_nodes)}

        # Until there is no change between our last guess and the next one
        while old_layers != node_layers:
            old_layers = node_layers
            for connection in connections:
                if connection.enabled:
                    # Update the index of the input node to the max between 1 or the current value at the index
                    node_layers[connection.input_node] = max(1, node_layers[connection.input_node])
                    # Update the index of the output node between what the input layer is +1 or the current value of the
                    # output node
                    node_layers[connection.output_node] = max(node_layers[connection.output_node],
                                                              (node_layers[connection.input_node] + 1))

        layer_nodes = {key: [] for key in node_layers.values()}
        for node_id, layer in node_layers.items():
            layer_nodes[layer].append(node_id)

        # The last layer should only contain the output node
        if len(layer_nodes[max(layer_nodes)]) != 1:
            raise Exception('Invalid genome has been unpacked')

        return node_layers, layer_nodes

    @classmethod
    def get_non_connected_source_node_positions(cls, nodes, node_map):
        """
        Gives a position to source nodes which don't have any connections
        :param nodes: A dictionary (node_id, node_gene)
        :param node_map: Keeps track of what position each node is within it's respective layer
        """
        # Get all the source nodes
        source_nodes = []
        for node in nodes.values():
            if node.node_type == 'source':
                source_nodes.append(node.node_id)

        # Find the max position for source nodes
        source_nodes_in_map = []
        source_nodes_not_in_map = []
        max_in_layer_position = None
        for source_node in source_nodes:
            if source_node in node_map:
                # Find the max position for the source nodes in that source node layer
                if max_in_layer_position is None or node_map[source_node] > max_in_layer_position:
                    max_in_layer_position = node_map[source_node]
                source_nodes_in_map.append(source_node)
            else:
                source_nodes_not_in_map.append(source_node)

        # Give all the source nodes not in the map a position value
        for node in source_nodes_not_in_map:
            max_in_layer_position += 1
            node_map[node] = max_in_layer_position

        return source_nodes_in_map

    @classmethod
    def get_node_map(cls, num_layers, layer_nodes, nodes):
        """
        :param nodes: All the nodes in the genome
        :param num_layers: Number of layers for the genome
        :param layer_nodes: A dictionary with a list of nodes for each layer
        :return: A dictionary containing for each node_id which number node they are in their respective layer
        """
        node_map = {}
        for layer in range(1, num_layers + 1):
            counter = 1
            # We go through each index of a node for each layer at assign it an incremented number.
            for node_id in layer_nodes[layer]:
                node_map[node_id] = counter
                counter += 1

        # Get the position for source nodes which aren't connected to anything
        source_nodes = cls.get_non_connected_source_node_positions(nodes=nodes, node_map=node_map)

        return node_map, source_nodes

    @classmethod
    def find_ghost_nodes(cls, nodes_per_layer, node_layers, node_map, connections, layer_nodes):
        """
        :param nodes_per_layer: Dictionary containing number of nodes per layer
        :param node_layers: Dictionary containing the
        :param node_map: Keeps track of which node is which number in it's respective layer
        :param connections: A list containing all the ConnectionGenes
        :param layer_nodes: Dictionary with nodes list for each layer
        :return: The new added added_nodes and added_connections
        """
        # Will save which new nodes have been added
        added_nodes = []
        # Will save which new connections have been added
        added_connections = []

        # This will contain the last dummy node in the connections: key: the input node, value: the last dummy node
        last_dummy_related_to_connection = {}

        for connection in connections:
            # How many layers the connection spans
            input_node_layer = node_layers[connection.input_node]
            output_node_layer = node_layers[connection.output_node]
            layer_span = output_node_layer - input_node_layer

            if layer_span > 1 and connection.enabled:
                # Remove the connection since it will be replaced
                connections.remove(connection)

                num_added_nodes = layer_span - 1
                new_nodes = cls.update_nodes(num_added_nodes=num_added_nodes, added_nodes=added_nodes,
                                             node_layers=node_layers, input_node_layer=input_node_layer,
                                             nodes_per_layer=nodes_per_layer, node_map=node_map,
                                             layer_nodes=layer_nodes)

                # Turn off the connection we're in
                connection.enabled = False
                cls.update_connections(new_node_ids=new_nodes, connection_gene=connection,
                                       added_connections=added_connections,
                                       connection_replaced_weight=connection.weight,
                                       last_dummy_related_to_connection=last_dummy_related_to_connection)

        return added_nodes, added_connections, last_dummy_related_to_connection

    @classmethod
    def update_nodes(cls, num_added_nodes, added_nodes, node_layers, input_node_layer, nodes_per_layer,
                     node_map, layer_nodes):
        """
        :param num_added_nodes: Number of nodes to be added
        :param added_nodes: The list of added nodes so far (so we can append the new ones we make in this function)
        :param node_layers: A dictionary containing the layer associated with each node_id
        :param input_node_layer: The starting input layer
        :param nodes_per_layer: A dictionary containing a count of the number of nodes in each layer
        :param node_map: A dictionary containing for each node_id which number node they are in their respective layer
        :param layer_nodes: A dictionary containing a list for each layer of the node_ids in the layer
        :return: A list of the new_node_ids created
        """
        # Get the highest value node_id because we will use this as the starting point for the next nodes
        last_node_id = max(node_layers.keys())
        # A list of what the node id's will be for the newly added nodes
        new_node_ids = list(range(last_node_id + 1, last_node_id + num_added_nodes + 1))
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
            # Update the the nodes for each layer
            layer_nodes[layer].append(node_id)

            # Create the node and give it a random bias
            new_node_gene = NodeGene(node_id=node_id, node_type='hidden', bias=np.random.randn())
            # Keep track of the new nodes we've added
            added_nodes.append(new_node_gene)

        return new_node_ids

    @classmethod
    def update_connections(cls, new_node_ids, connection_gene, added_connections, connection_replaced_weight,
                           last_dummy_related_to_connection):
        """
        :param connection_replaced_weight: The weight of the connection being replaced
        :param new_node_ids: The newly added nodes
        :param connection_gene: The connection gene class object
        :param added_connections: A List of added connections so far
        :return: None
        """
        new_input_nodes = [connection_gene.input_node] + new_node_ids
        new_output_nodes = new_node_ids + [connection_gene.output_node]
        num_new_connections = len(new_input_nodes)

        # Set the last input node as the related dummy to the connection input
        last_dummy_related_to_connection[(connection_gene.input_node, connection_gene.output_node)] = new_input_nodes[
            len(new_input_nodes) - 1]

        for new_connection_number in range(num_new_connections):
            # Create the new connection gene with a random weight
            new_connection_gene = ConnectionGene(input_node=new_input_nodes[new_connection_number],
                                                 output_node=new_output_nodes[new_connection_number],
                                                 # Only the last connection of the new connections should have the replaced weight
                                                 weight=connection_replaced_weight if new_connection_number == (
                                                         num_new_connections - 1) else np.random.randn())
            # num_new_connections - 1 because of python indexing.
            if new_connection_number != (num_new_connections - 1):
                # Because all the weights apart from the last on to connect to the final node should have constant 1
                # weights
                new_connection_gene.keep_constant_weight = True

            # Keep track of the new connections we've added
            added_connections.append(new_connection_gene)

    @classmethod
    def connections_matrix(cls, layer_connections, num_inputs, num_outputs, node_map):
        """
        :param layer_connections: A list containing the connections ONLY for the specific layer
        :param num_inputs: Number of inputs for the layer
        :param num_outputs: Number of outputs for the layer
        :param node_map: Dictionary with respective position in a layer for each node_id
        :return: numpy array of which nodes are connected to which, a bias matrix which should be multiplied with the
                actual bias matrix so that the constant weight connections don't have bias applied to them. And  a list
                of connection genes for which the weight should remain a constant one.
        """
        connection_matrix = np.zeros((num_inputs, num_outputs))
        # Keeps track of which connections shouldn't have any bias or activation functions applied
        bias_matrix = np.ones((1, num_outputs))
        # Keep track of which connections should keep a constant one weight
        keep_constant_weight_connections = []
        for connection in layer_connections:
            # Double check that the connection is enabled
            if connection.enabled:
                # Minus one because of python indexing
                input_position_within_layer = node_map[connection.input_node] - 1
                output_position_within_layer = node_map[connection.output_node] - 1
                # Set it to one if it exists
                connection_matrix[input_position_within_layer, output_position_within_layer] = 1
                # Keep track of which connections should be a constant weight i.e. weight of 1 forever
                if connection.keep_constant_weight:
                    # Because when we multiply by the actual biases, then we will remove the on that should have a
                    # constant weight
                    bias_matrix[0, output_position_within_layer] = 0
                    # Saving the connections because we have to always set the weight to one for that specific position.
                    # It will always be used to ensure that the activation function isn't applied to that value.
                    keep_constant_weight_connections.append(connection)
        return connection_matrix, bias_matrix, keep_constant_weight_connections

    @classmethod
    def all_connection_matrices(cls, connections, layer_nodes, node_layers, num_layers, node_map):
        """
        :param connections: List of connection genes
        :param layer_nodes:  Nodes for each layer
        :param node_layers: Layer associated with each node_id
        :param num_layers: Number of layers in genome
        :param node_map: Position for each node_id in it's respective layer
        :return: A dictionary containing whether a node is connected a node in the next layer for every layer, a bias matrix
        for each layer indicating which nodes shouldn't have a bias applied, and a list of the connections which should
        have a constant 1 weight connection
        """
        connection_matrices = {}
        bias_matrices = {}
        constant_weight_connections = {}
        layer_connections_dict = {}
        for layer in range(1, num_layers):
            num_inputs = len(layer_nodes[layer])
            num_outputs = len(layer_nodes[layer + 1])
            layer_connections = []
            for connection in connections:
                # The layer a connection is associated with depends on which layer the output node is on. Layer + 1
                # because we're only counting the layers excluding the data input layer.
                if node_layers[connection.output_node] == layer + 1:
                    layer_connections.append(connection)

            layer_connections_dict[layer] = layer_connections
            connection_matrices[layer], bias_matrices[layer], constant_weight_connections[
                layer] = cls.connections_matrix(
                layer_connections=layer_connections,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                node_map=node_map)
        return connection_matrices, bias_matrices, constant_weight_connections, layer_connections_dict


def main():
    from genome import Genome
    node_list = [NodeGene(node_id=1, node_type='source'),
                 NodeGene(node_id=2, node_type='source'),
                 NodeGene(node_id=3, node_type='hidden'),
                 NodeGene(node_id=4, node_type='hidden'),
                 NodeGene(node_id=5, node_type='output')]

    connection_list = [ConnectionGene(input_node=1, output_node=5, innovation_number=1, enabled=True),
                       ConnectionGene(input_node=1, output_node=4, innovation_number=2, enabled=True),
                       ConnectionGene(input_node=2, output_node=3, innovation_number=3, enabled=True),
                       ConnectionGene(input_node=2, output_node=4, innovation_number=4, enabled=True),
                       ConnectionGene(input_node=3, output_node=5, innovation_number=5, enabled=True),
                       ConnectionGene(input_node=4, output_node=5, innovation_number=6, enabled=True)]

    genome = Genome(nodes=node_list, connections=connection_list, key=1)

    counter = 0
    for x in (DeconstructGenome.unpack_genome(genome)):
        if counter == 0:
            print(max(x))
        else:
            print(x)
        counter += 1


if __name__ == "__main__":
    main()
