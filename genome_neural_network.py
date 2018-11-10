from deconstruct_genome import DeconstructGenome
from neural_network import *
from genome import *
from gene import *
from neural_network_components import *


class GenomeNeuralNetwork:

    def __init__(self, genome, x_train, y_train, create_weights_bias_from_genome, learning_rate=0.0001,
                 num_epochs=1000, batch_size=64):
        self.genome = genome

        # Unpack genome
        self.connection_matrices_per_layer, self.no_activations_matrix_per_layer, self.constant_weight_connections, \
        self.nodes_per_layer, self.node_map, self.layer_connections_dict, self.updated_nodes = DeconstructGenome.unpack_genome(
            genome=genome)

        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        # Where weights are saved
        self.weights_dict = dict()
        # Where bias for every layer is saved
        self.bias_dict = dict()
        # Dictionary of activation functions for each layer
        self.activation_function_dict = dict()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        # Minus one because we include the data input as a layer in 'nodes_per_layer'
        self.num_layers = max(self.nodes_per_layer) - 1

        # The number of 'nodes' on the first layer should be equal to the number of features in the training data
        assert (self.nodes_per_layer[1] == self.x_train.shape[1])

        # Set the activation functions for each layer
        self.initialise_activation_functions(activation_type='sigmoid')

        self.initialise_parameters(have_bias=True, create_weight_bias_matrix_from_genome=create_weights_bias_from_genome)

        # This is to check that there is an activation function specified for each layer
        assert (len(self.connection_matrices_per_layer) == len(self.activation_function_dict))

        # The activation function for the last layer should be a sigmoid due to how the gradients were calculated
        assert (self.activation_function_dict[len(self.connection_matrices_per_layer)] == ActivationFunctions.sigmoid)

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
        for layer in range(1, self.num_layers + 1):
            for connection in self.layer_connections_dict[layer]:
                # Get their relative position inside their respective layer
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

    def run_one_pass(self, input_data, labels):
        """
        One pass counts as one forward propagation and one backware propogation including the optimisation of the
        paramters
        :return: The cost for the current step
        """

        n_examples = input_data.shape[0]

        prediction, layer_input_dict = ForwardProp.genome_forward_prop(num_layers=self.num_layers,
                                                                       initial_input=input_data,
                                                                       layer_weights=self.weights_dict,
                                                                       layer_biases=self.bias_dict,
                                                                       layer_activation_functions=self.activation_function_dict,
                                                                       keep_constant_connections=self.constant_weight_connections,
                                                                       node_map=self.node_map)

        # Asserting that the prediction gives the same number of outputs as expected
        assert (labels.shape[0] == prediction.shape[0])

        # Excluded bias gradients here
        weight_gradients, bias_gradients = BackProp.back_prop(num_layers=self.num_layers,
                                                              layer_inputs=layer_input_dict,
                                                              layer_weights=self.weights_dict,
                                                              layer_activation_functions=self.activation_function_dict,
                                                              expected_y=labels, predicted_y=prediction)

        self.optimise_parameters(weight_gradients=weight_gradients, bias_gradients=bias_gradients)

        # Define cost function
        loss = -((labels * np.log(prediction)) + ((1 - labels) * np.log(1 - prediction)))
        cost = (1 / n_examples) * np.sum(loss + 1e-8, axis=0)

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

    def optimise(self, error_stop=None):
        """
        Train the neural network
        :return: a list of each epoch with the cost associated with it
        """

        epoch_list = list()
        cost_list = list()

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

            print('EPOCH:', epoch, 'Cost:', round(epoch_cost, 3))

        return epoch_list, cost_list


def main():
    optimise = True

    node_list = [NodeGene(node_id=1, node_type='source'),
                 NodeGene(node_id=2, node_type='source'),
                 NodeGene(node_id=3, node_type='hidden', bias=0.5),
                 NodeGene(node_id=4, node_type='hidden', bias=-1.5),
                 NodeGene(node_id=5, node_type='output', bias=1.5)]

    connection_list = [ConnectionGene(input_node=1, output_node=3, innovation_number=1, enabled=True, weight=1),
                       ConnectionGene(input_node=1, output_node=4, innovation_number=2, enabled=True, weight=-1),
                       ConnectionGene(input_node=2, output_node=3, innovation_number=3, enabled=True, weight=1),
                       ConnectionGene(input_node=2, output_node=4, innovation_number=4, enabled=True, weight=-1),
                       ConnectionGene(input_node=3, output_node=5, innovation_number=5, enabled=True, weight=1),
                       ConnectionGene(input_node=4, output_node=5, innovation_number=6, enabled=True, weight=1)]

    genome = Genome(nodes=node_list, connections=connection_list, key=1)

    # Test and Train data
    data_train, labels_train = create_data(n_generated=5000)

    genome_nn = GenomeNeuralNetwork(genome=genome, x_train=data_train, y_train=labels_train, learning_rate=0.1, create_weights_bias_from_genome=False)

    if optimise:
        epochs, cost = genome_nn.optimise(error_stop=0.09)

        plt.plot(epochs, cost)
        plt.show()


if __name__ == "__main__":
    main()
