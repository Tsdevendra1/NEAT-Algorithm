from deconstruct_genome import DeconstructGenome
from neural_network import *
from genome import *
from gene import *
from neural_network_components import *


class GenomeNeuralNetwork:

    def __init__(self, genome, x_train, y_train, create_weights_bias_from_genome, activation_type, learning_rate=0.0001,
                 num_epochs=1000, batch_size=64, show_connections=False):
        """
        :param genome: The genome class from which the the neural network will be made from
        :param x_train: The x_training data in the shape (num_examples, num_features)
        :param y_train: The y_training data in the shape (num_examples, 1) only one column indicating which category
        :param create_weights_bias_from_genome: True or False if you want the neural network to initialise with the weights
        from the genome connections and the biases from the genome nodes
        :param activation_type: Specify the activation type for each layer excluding the last layer. This will be used
        for every layer apart from the last.
        :param learning_rate: Learning rate
        :param num_epochs: Number of iterations to run
        :param batch_size: Batch size since it is coded with stochastic gradient descent
        :param show_connections: True or false if the connections for the neural network AFTER it has been unpack from
        the genome are needed to be shown
        """
        self.genome = genome
        # This is a matrix containing 1's and 0's depending on if there is a connection or not for each weight matrix
        # in the neural network
        self.connection_matrices_per_layer = genome.connection_matrices_per_layer
        # Similar to above, contains a matrix for the biases for which nodes should never have a bias applied (due to
        # being a dummy node)
        self.no_activations_matrix_per_layer = genome.no_activations_matrix_per_layer
        # A list of connections which will always have a constant 1 connection value
        self.constant_weight_connections = genome.constant_weight_connections
        # The number of nodes for each layer
        self.nodes_per_layer = genome.nodes_per_layer
        # A dictionary with which position a node is in for it's respective layer
        self.node_map = genome.node_map
        # Dictionary with the connections for each layer (so weight matrix can be made)
        self.layer_connections_dict = genome.layer_connections_dict
        # An updated list of the nodes. Contains dummys nodes as well.
        self.updated_nodes = genome.updated_nodes
        # A dictionary with the nodes for each layer
        self.layer_nodes = genome.layer_nodes
        # A dict containing which layer each node is on
        self.node_layers = genome.node_layers
        # Key: input_node for a connection that spans multiple layers, value: the last dummy node of the set of connections to reach the final node in the connection
        self.last_dummy_related_to_connection = genome.last_dummy_related_to_connection

        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        # Where weights are saved
        self.weights_dict = {}
        # Where bias for every layer is saved
        self.bias_dict = {}
        # Dictionary of activation functions for each layer
        self.activation_function_dict = {}
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        # Minus one because we include the data input as a layer in 'nodes_per_layer'
        self.num_layers = max(self.nodes_per_layer) - 1

        # So far there have only been two implementations for the activation type
        assert (activation_type in ('relu', 'sigmoid'))

        # Set the activation functions for each layer
        self.initialise_activation_functions(activation_type=activation_type)

        self.initialise_parameters(have_bias=True,
                                   create_weight_bias_matrix_from_genome=create_weights_bias_from_genome)

        self.x_train = self.modify_x_data_for_sources()

        # This is to check that there is an activation function specified for each layer
        assert (len(self.connection_matrices_per_layer) == len(self.activation_function_dict))

        # The activation function for the last layer should be a sigmoid due to how the gradients were calculated
        assert (self.activation_function_dict[len(self.connection_matrices_per_layer)] == ActivationFunctions.sigmoid)

        if self.x_train.shape[1] != self.weights_dict[1].shape[0]:
            raise Exception('The training data has not been correctly modified to fit the connections')

        if show_connections:
            for key, value in self.layer_connections_dict.items():
                print('Layer: {}'.format(key), 'Connections: {}'.format(value), '\n')

    def initialise_activation_functions(self, activation_type):
        for layer in range(1, self.num_layers):
            self.activation_function_dict[
                layer] = ActivationFunctions.relu if activation_type == 'relu' else ActivationFunctions.sigmoid

        # The last activation should always be a sigmoid because that's how the gradients are calculated
        self.activation_function_dict[self.num_layers] = ActivationFunctions.sigmoid

    def ensure_correct_weights(self):
        """
        Goes through the weights dict to ensure the where there are no connections the weight is set to zero, and where
        the connection should be a constant 1, that it is set as such.
        """
        for layer in self.weights_dict:
            # Zeroes out the connections which there isn't one
            self.weights_dict[layer] *= self.connection_matrices_per_layer[layer]
            # All the connections where the connection should have a constant one connection are set to one
            for connection in self.constant_weight_connections[layer]:
                # Need to convert to their position in their layer
                # Minus one because of python indexing
                input_position_within_layer = self.node_map[connection.input_node] - 1
                output_position_within_layer = self.node_map[connection.output_node] - 1
                # Set the weight to one
                self.weights_dict[layer][input_position_within_layer, output_position_within_layer] = 1

    def ensure_correct_bias(self):
        """
        Zeroes out the bias values where the node is a dummy node
        """
        for layer in range(1, self.num_layers + 1):
            # Broadcast to fit the batch_size and also multiply by the matrix which contains which bias values should
            #  be zero
            self.bias_dict[layer] *= self.no_activations_matrix_per_layer[layer]

    @staticmethod
    def xavier_initalizer(num_inputs, num_outputs):
        """
        NOTE: if using RELU then use constant 2 instead of 1 for sqrt
        """
        np.random.seed(7)
        weights = np.random.randn(num_inputs, num_outputs) * np.sqrt(2 / num_inputs)

        return weights

    def create_weight_bias_matrix_from_genome(self):
        # connections_dict_by_nodes = {}
        # for connection in self.genome.connections.values():
        #     connections_dict_by_nodes[(connection.input_node, connection.output_node)] = connection
        #
        # try:
        #     for layer in range(1, self.num_layers + 1):
        #         for layer_connection in self.layer_connections_dict[layer]:
        #             connection = connections_dict_by_nodes[(layer_connection.input_node, layer_connection.output_node)]
        #             # connection = connections_dict_by_nodes.get((layer_connection.input_node, layer_connection.output_node))
        #             # if connection and connection.enabled:
        #             if connection.enabled:
        #                 # They both need to be set if we're going to create the weight matrix from them
        #                 if connection.weight is None:
        #                     raise ValueError('You have not set a weight for this connection')
        #                 if self.updated_nodes[connection.output_node].bias is None:
        #                     raise ValueError('The node doesnt have a bias value, please set one in the nodes list')
        #
        #                 # Get their relative position inside their respective layer. Minus one for python indexing
        #                 input_node_position = self.node_map[connection.input_node] - 1
        #                 output_node_position = self.node_map[connection.output_node] - 1
        #                 self.weights_dict[layer][input_node_position, output_node_position] = connection.weight
        #                 self.bias_dict[layer][0, output_node_position] = self.genome.nodes[connection.output_node].bias
        # except:
        #     raise Exception('Error in initialising weight and bias matrices')

        for layer in range(1, self.num_layers + 1):
            for connection in self.layer_connections_dict[layer]:
                if connection.enabled:
                    # They both need to be set if we're going to create the weight matrix from them
                    if connection.weight is None:
                        raise ValueError('You have not set a weight for this connection')
                    if self.updated_nodes[connection.output_node].bias is None:
                        raise ValueError('The node doesnt have a bias value, please set one in the nodes list')

                    # Get their relative position inside their respective layer. Minus one for python indexing
                    input_node_position = self.node_map[connection.input_node] - 1
                    output_node_position = self.node_map[connection.output_node] - 1
                    self.weights_dict[layer][input_node_position, output_node_position] = connection.weight
                    self.bias_dict[layer][0, output_node_position] = self.updated_nodes[connection.output_node].bias

    def initialise_parameters(self, create_weight_bias_matrix_from_genome, have_bias=False):
        """
        :param create_weight_bias_matrix_from_genome: A boolean of whether we should use the weights already given to each gene
        :param have_bias: Indicates whether to intialise a bias parameter as well
        """
        # Initialise parameters
        for layer in range(1, self.num_layers + 1):

            # We multiply by the connection_matrices_per_layer because zeroes out the weights where there isn't a
            # connection between the nodes. Add one because nodes per_layer counts the first layer as the data_inputs.
            # So indexing nodes_per_layer[1] would actually give you the number of features in the training set.
            self.weights_dict[layer] = self.xavier_initalizer(
                num_inputs=self.nodes_per_layer[layer],
                num_outputs=self.nodes_per_layer[layer + 1])

            if have_bias:
                # Shape is (1, num_outputs) for the layer
                self.bias_dict[layer] = np.zeros((1, self.nodes_per_layer[layer + 1]))

        if create_weight_bias_matrix_from_genome:
            self.create_weight_bias_matrix_from_genome()

        # Zeroes out connections where there aren't any and sets the constant connections to one
        self.ensure_correct_weights()

        # Zeroes out the bias values for each layer where there is a dummy node
        self.ensure_correct_bias()

    def run_one_pass(self, input_data, labels=None, return_cost_only=False, return_prediction_only=False):
        """
        One pass counts as one forward propagation and one backware propogation including the optimisation of the
        paramters
        :param return_prediction_only: Boolean on whether just to return the predictions
        :param labels: The correct labels
        :param input_data: the input data used to train
        :type return_cost_only: True or false of it you just want the cost instead of optimising as well
        :return: The cost for the current step
        """

        n_examples = input_data.shape[0]

        # We have to modify the input data incase one of the features isn't present in the genome connections (i.e. it got removed due to a mutation)
        # This automatically gets done to the data which is passed to the GenomeNeuralNetwork initialisation, but since
        # we allow custom input data for this function we must do the check again
        if self.x_train.shape[1] != input_data.shape[1]:
            input_data = self.modify_x_data_for_sources(x_data=input_data)

        prediction, layer_input_dict = ForwardProp.genome_forward_prop(num_layers=self.num_layers,
                                                                       initial_input=input_data,
                                                                       layer_weights=self.weights_dict,
                                                                       layer_biases=self.bias_dict,
                                                                       layer_activation_functions=self.activation_function_dict,
                                                                       keep_constant_connections=self.constant_weight_connections,
                                                                       node_map=self.node_map)

        if return_prediction_only:
            return prediction

        # Because labels is optional, if the function hasn't already returned, labels should exist.
        assert (labels is not None)
        # Asserting that the prediction gives the same number of outputs as expected
        assert (labels.shape[0] == prediction.shape[0])

        # Define cost function
        loss = -((labels * np.log(prediction)) + ((1 - labels) * np.log(1 - prediction)))
        cost = (1 / n_examples) * np.sum(loss + 1e-8, axis=0)

        # We optimise in the case where we're not just interested in the cost
        if not return_cost_only:
            # Excluded bias gradients here
            weight_gradients, bias_gradients = BackProp.back_prop(num_layers=self.num_layers,
                                                                  layer_inputs=layer_input_dict,
                                                                  layer_weights=self.weights_dict,
                                                                  layer_activation_functions=self.activation_function_dict,
                                                                  expected_y=labels, predicted_y=prediction)

            self.optimise_parameters(weight_gradients=weight_gradients, bias_gradients=bias_gradients)

        return cost[0]

    def optimise_parameters(self, weight_gradients, bias_gradients=None):
        """
        :param weight_gradients: Dictionary containing weight gradients for each layer
        :param bias_gradients: Dictionary containing bias gradients for each layer
        """

        for layer_number in weight_gradients:
            self.weights_dict[layer_number] = self.weights_dict[layer_number] - (
                    self.learning_rate * weight_gradients[layer_number])

            if bias_gradients is not None:
                self.bias_dict[layer_number] = self.bias_dict[layer_number] - (
                        self.learning_rate * bias_gradients[layer_number])

        # Zeroes out connections where there aren't any and sets the constant connections to one
        self.ensure_correct_weights()

        # Zeroes out the bias values for each layer where there is a dummy node
        self.ensure_correct_bias()

    def update_genes(self):
        """
        Here we are updating all the connection weights with the optimised weights, after the neural network for the
        genome has run
        :return:
        """
        for connection in self.genome.connections.values():
            if connection.enabled:
                input_node_layer = self.node_layers[connection.input_node]
                output_node_layer = self.node_layers[connection.output_node]
                layer_span = output_node_layer - input_node_layer

                if layer_span == 1:
                    # Minus one because of python indexing
                    input_node_position = self.node_map[connection.input_node] - 1
                    output_node_position = self.node_map[connection.output_node] - 1

                    connection_weight = self.weights_dict[output_node_layer - 1][
                        input_node_position, output_node_position]

                    # TODO: Improve implementation
                    # I'm not sure why I have layer connections and the connections in a genome as seperate. They should be pointing to the same connection instance.
                    # This is just a workaround for now.
                    for layer_connection in self.genome.layer_connections_dict[input_node_layer]:
                        if (layer_connection.input_node, layer_connection.output_node) == (
                                connection.input_node, connection.output_node):
                            layer_connection.weight = connection_weight
                            break

                    # Minus one because node_layers counts the first layer as a layer where as the weights dict doesn't
                    connection.weight = connection_weight
                # If the connection spans multiple layers
                else:
                    try:
                        # Start at one because layers start at one
                        last_dummy_node = self.last_dummy_related_to_connection[
                            (connection.input_node, connection.output_node)]

                        # Minus one because of python indexing
                        last_dummy_node_position = self.node_map[last_dummy_node] - 1
                        # This is the position within the layer
                        output_node_position = self.node_map[connection.output_node] - 1

                        last_dummy_node_layer = self.node_layers[last_dummy_node]

                        connection_weight = self.weights_dict[output_node_layer - 1][
                            last_dummy_node_position, output_node_position]

                        # TODO: Improve implementation
                        # I'm not sure why I have layer connections and the connections in a genome as seperate. They should be pointing to the same connection instance.
                        # This is just a workaround for now.
                        for layer_connection in self.genome.layer_connections_dict[last_dummy_node_layer]:
                            if (layer_connection.input_node, layer_connection.output_node) == (
                                    last_dummy_node, connection.output_node):
                                layer_connection.weight = connection_weight
                                break

                        # Minus one because node_layers counts the first layer as a layer where as the weights dict doesn't
                        connection.weight = connection_weight
                    except KeyError as e:
                        raise Exception(e)

        for node in self.genome.nodes.values():
            node_layer = self.node_layers.get(node.node_id)
            # Can't modify the bias of a source node and if the node_layer is not found it means the connection it is
            #  related to is not enabled
            if node.node_type != 'source' and node_layer:
                # These should be the same value since we're updating two instances of it due to poor implementation on my part
                updated_node = self.genome.updated_nodes[node.node_id]
                # Minus one for indexing
                node_position = self.node_map[node.node_id] - 1
                # Minus one because node_layers counts the first layer as a layer where as the bias dict doesn't
                bias_value = self.bias_dict[node_layer - 1][0, node_position]
                updated_node.bias = bias_value
                node.bias = bias_value

    def optimise(self, print_epoch, error_stop=None):
        """
        Train the neural network
        :param print_epoch: Boolean, print the cost every epoch or not
        :param error_stop: whether to stop if a certange error threshold is reached
        :return: a list of each epoch with the cost associated with it
        """

        epoch_list = []
        cost_list = []

        for epoch in range(self.num_epochs):

            for batch_start in range(0, self.x_train.shape[0], self.batch_size):
                current_batch = self.x_train[batch_start:batch_start + self.batch_size, :]
                current_labels = self.y_train[batch_start:batch_start + self.batch_size, :]

                epoch_cost = self.run_one_pass(input_data=current_batch, labels=current_labels)

            epoch_list.append(epoch)
            cost_list.append(epoch_cost)

            # Finish early if it is optimised to a certain error
            if error_stop and epoch_cost < error_stop:
                break

            # Print progress every 10%
            # if epoch % (self.num_epochs * 0.1) == 0 and epoch != 0:
            #     percentage_complete = (epoch / self.num_epochs) * 100
            #     print('Optimising {}% done...'.format(percentage_complete))

            if print_epoch:
                print('EPOCH:', epoch, 'Cost:', round(epoch_cost, 3))

            first_cost = cost_list[0]
            # Check progress quarter of the way through
            if epoch == (self.num_epochs // 4):
                quarter_cost = cost_list[self.num_epochs // 4]
                percentage_increase_quarter = (1 - quarter_cost / first_cost) * 100
                print('Beginning cost: {}'.format(round(first_cost, 3)))
                print('Quarter cost: {}'.format(round(quarter_cost, 3)))
                print(
                    'Percentage increase from beginning to quarter of the way: {}'.format(
                        round(percentage_increase_quarter, 1)))
                if percentage_increase_quarter < 1:
                    print('STOPPING OPTIMISATION DUE TO POOR PERFORMANCE')
                    break

            # Check progress halfway through
            if epoch == (self.num_epochs // 2):
                midway_cost = cost_list[self.num_epochs // 2]
                percentage_increase_midway = (1 - midway_cost / first_cost) * 100
                percentage_increase_between_quarter_and_midway = (1 - midway_cost / quarter_cost) * 100
                print('Midway cost: {}'.format(round(midway_cost, 3)))
                print('Percentage increase from beginning to halfway: {}'.format(round(percentage_increase_midway, 1)))
                print('Percentage increase from quarter to halfway: {}'.format(
                    round(percentage_increase_between_quarter_and_midway, 1)))
                if percentage_increase_quarter == percentage_increase_midway \
                        or percentage_increase_quarter > percentage_increase_midway \
                        or percentage_increase_between_quarter_and_midway < 1:
                    print('STOPPING OPTIMISATION DUE TO POOR PERFORMANCE')
                    break

            # Check progress 3/4 of the way
            if epoch == (self.num_epochs * 3 // 4):
                three_quarters_cost = cost_list[self.num_epochs * 3 // 4]
                percentage_increase_three_quarters = (1 - three_quarters_cost / first_cost) * 100
                percentage_increase_between_midway_and_three_quarters = (1 - three_quarters_cost / midway_cost) * 100
                print('Three quarters cost: {}'.format(round(three_quarters_cost, 3)))
                print('Percentage increase from beginning to halfway: {}'.format(
                    round(percentage_increase_three_quarters, 1)))
                print('Percentage increase from midway to three quarters: {}'.format(
                    round(percentage_increase_between_midway_and_three_quarters, 1)))
                if percentage_increase_midway == percentage_increase_three_quarters \
                        or percentage_increase_midway > percentage_increase_three_quarters \
                        or percentage_increase_between_midway_and_three_quarters < 1:
                    print('STOPPING OPTIMISATION DUE TO POOR PERFORMANCE')
                    break

            # # If the cost is getting worse we stop the algorithm
            # if cost_list[epoch] > cost_list[epoch-1]:
            #     print('STOPPING OPTIMISATION DUE TO POOR PERFORMANCE COST GETTING WORSE')
            #     break

        print('Optimising 100% done...')

        self.update_genes()

        return epoch_list, cost_list

    def modify_x_data_for_sources(self, x_data=None):
        """
        Since not every genome will have a connection for all the source nodes, we must account for this in the
        training data by removing features if there isn't a connection to it.
        """

        # x_data = self.x_train
        if x_data is None:
            x_data = self.x_train

        source_nodes = {}

        # Find the source nodes in the genome
        for node in self.genome.nodes.values():
            if node.node_type == 'source':
                if node not in source_nodes:
                    source_nodes[node.node_id] = 0

        # Check how many connections there are for each source node
        for connection in self.genome.connections.values():
            if connection.enabled:
                if connection.input_node in source_nodes:
                    source_nodes[connection.input_node] += 1

        not_connection_sources = set()
        has_connection_sources = set()
        # Filter the one's which don't have any connections
        for node_id, connections_amount in source_nodes.items():
            if connections_amount == 0:
                not_connection_sources.add(node_id)
            else:
                has_connection_sources.add(node_id)

        # Keeps track of which positions (i.e. columns) will be deleted from x_data
        node_positions_to_delete = []

        # Account for the fact that a source node might be deleted
        if len(source_nodes) != x_data.shape[1]:
            possible_positions = set(range(x_data.shape[1]))
            included_position_by_node = set()
            for node_id in has_connection_sources:
                included_position_by_node.add(self.node_map[node_id] - 1)

            for node_position in possible_positions:
                if node_position not in included_position_by_node:
                    node_positions_to_delete.append(node_position)

        # Delete the columns for the source nodes which don't have connections
        for node in not_connection_sources:
            # Get the position of the node inside it's own layer. Minus 1 for python indexing
            node_position = self.node_map[node] - 1
            node_positions_to_delete.append(node_position)

        # Remove duplicated
        node_positions_to_delete = list(set(node_positions_to_delete))
        # Put the highest position first, and delete it first so that if there is more than one needed to be deleted,
        #  the index is not now messed up due to previous deletion
        node_positions_to_delete.sort(reverse=True)
        for node_position in node_positions_to_delete:
            if node_position < x_data.shape[1]:
                x_data = np.delete(x_data, node_position, axis=1)


        return x_data


def main():
    node_list = [NodeGene(node_id=1, node_type='source'),
                 NodeGene(node_id=2, node_type='source'),
                 NodeGene(node_id=3, node_type='hidden', bias=0.5),
                 NodeGene(node_id=4, node_type='hidden', bias=-1.5),
                 NodeGene(node_id=5, node_type='output', bias=1.5)]

    connection_list = [ConnectionGene(input_node=1, output_node=5, innovation_number=1, enabled=True, weight=9),
                       ConnectionGene(input_node=1, output_node=3, innovation_number=2, enabled=True, weight=2),
                       ConnectionGene(input_node=2, output_node=3, innovation_number=3, enabled=True, weight=3),
                       ConnectionGene(input_node=2, output_node=4, innovation_number=4, enabled=True, weight=4),
                       ConnectionGene(input_node=2, output_node=5, innovation_number=5, enabled=True, weight=3),
                       ConnectionGene(input_node=3, output_node=5, innovation_number=6, enabled=True, weight=5),
                       ConnectionGene(input_node=4, output_node=5, innovation_number=7, enabled=True, weight=7)]

    genome = Genome(nodes=node_list, connections=connection_list, key=1)

    # Test and Train data
    data_train, labels_train = create_data(n_generated=5000)

    genome_nn = GenomeNeuralNetwork(genome=genome, x_train=data_train, y_train=labels_train, learning_rate=0.1,
                                    create_weights_bias_from_genome=False, activation_type='sigmoid',
                                    show_connections=True)

    optimise = False

    if optimise:
        epochs, cost = genome_nn.optimise(error_stop=0.09)

        plt.plot(epochs, cost)
        plt.xlabel('Epoch Number')
        plt.ylabel('Error')
        plt.title('Error vs Epoch Number')
        plt.show()


if __name__ == "__main__":
    main()
