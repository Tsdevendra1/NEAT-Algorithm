import numpy as np
import matplotlib.pyplot as plt
from neural_network_components import *


class NeuralNetwork:

    def __init__(self, x_train, y_train, layer_sizes, activation_function_dict, learning_rate=0.0001,
                 num_epochs=1000, batch_size=64):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.weights_dict = {}
        self.bias_dict = {}
        self.layer_sizes = layer_sizes
        self.activation_function_dict = activation_function_dict
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_layers = len(self.layer_sizes) - 1

        # Layer sizes should be a list with number of hidden nodes per layer. First number in list should be number of
        # features. And last number should always be one because there is one output.
        assert (layer_sizes[len(layer_sizes) - 1] == 1)
        assert (layer_sizes[0] == x_train.shape[1])

        # This is to check that there is an activation function specified for each layer
        assert (len(layer_sizes[1:len(layer_sizes)]) == len(activation_function_dict))

        # The activation function for the last layer should be a sigmoid due to how the gradients were calculated
        assert (activation_function_dict[len(layer_sizes) - 1] == ActivationFunctions.sigmoid)

        self.initialise_parameters(have_bias=True)

    @staticmethod
    def xavier_initalizer(num_inputs, num_outputs):
        """
        NOTE: if using RELU then use constant 2 instead of 1 for sqrt
        """
        np.random.seed(7)
        weights = np.random.randn(num_inputs, num_outputs) * np.sqrt(1 / num_inputs)

        return weights

    def initialise_parameters(self, have_bias=False):
        """
        :param have_bias: Indicates whether to intialise a bias parameter as well
        """
        # Initialise parameters
        for index in range(1, self.num_layers + 1):
            # Index +1 because we want layer_numbers to start at 1. Index -1 because number of inputs is number of
            # features from last layer.
            self.weights_dict[index] = self.xavier_initalizer(num_inputs=self.layer_sizes[index - 1],
                                                              num_outputs=self.layer_sizes[index])

            if have_bias:
                # Shape is (1, num_outputs) for the layer
                self.bias_dict[index] = np.zeros((1, self.layer_sizes[index]))

    def run_one_pass(self, input_data, labels):
        """
        One pass counts as one forward propagation and one backware propogation including the optimisation of the
        paramters
        :return: The cost for the current step
        """

        n_examples = input_data.shape[0]

        prediction, layer_input_dict = ForwardProp.forward_prop(num_layers=self.num_layers, initial_input=input_data,
                                                                layer_weights=self.weights_dict,
                                                                layer_activation_functions=self.activation_function_dict)

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

    def optimise(self, print_epoch_cost, error_stop=None):
        """
        Train the neural network
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

            if print_epoch_cost:
                print('EPOCH:', epoch, 'Cost:', round(epoch_cost, 3))

        return epoch_list, cost_list


def create_architecture(num_features_training, hidden_nodes_per_layer):
    assert (isinstance(hidden_nodes_per_layer, list))
    # See NeuralNetwork class for reasoning for this layout
    return [num_features_training] + hidden_nodes_per_layer + [1]


def create_data(n_generated):
    x_data = np.random.randint(2, size=(n_generated, 2))
    y_data = np.empty((n_generated, 1))

    # Sets y data to 1 or 0 according to XOR rules
    for column in range(x_data.shape[0]):
        y_data[column] = ((x_data[column, 0] == 1 and x_data[column, 1] == 1) or (
                x_data[column, 0] == 0 and x_data[column, 1] == 0))

    return x_data, y_data


def main():
    # Test and Train data
    data_train, labels_train = create_data(n_generated=5000)

    num_features = data_train.shape[1]

    #  This means it will be a two layer neural network with one layer being hidden with 2 nodes
    desired_architecture = [2, 2]
    nn_architecture = create_architecture(num_features, desired_architecture)

    # Defines the activation functions used for each layer
    activations_dict = {1: ActivationFunctions.relu, 2:ActivationFunctions.relu, 3: ActivationFunctions.sigmoid}

    neural_network = NeuralNetwork(x_train=data_train, y_train=labels_train, layer_sizes=nn_architecture,
                                   activation_function_dict=activations_dict, learning_rate=0.1, num_epochs=1000)

    epochs, cost = neural_network.optimise(error_stop=0.09, print_epoch_cost=True)

    plt.plot(epochs, cost)
    plt.show()


if __name__ == '__main__':
    main()
