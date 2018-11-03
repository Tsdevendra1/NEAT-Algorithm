import numpy as np
import matplotlib.pyplot as plt


class ActivationFunctions:

    @staticmethod
    def relu(input_matrix):
        output = np.maximum(input_matrix, 0, input_matrix)

        return output

    @staticmethod
    def sigmoid(x):
        activation = 1 / (1 + np.exp(-x))

        return activation

    @staticmethod
    def sigmoid_gradient(a):
        gradient = (a * (1 - a))

        return gradient

    @staticmethod
    def relu_gradient(x):
        gradient = x + 1e-8
        gradient[gradient < 0] = 0
        gradient[gradient > 1] = 1

        return gradient

    @staticmethod
    def get_activation_gradient(activation_function):
        """
        :param activation_function: A function, of the the activation function
        :return: the correct function to calculate the gradient
        """

        if activation_function == ActivationFunctions.relu:
            return ActivationFunctions.relu_gradient
        elif activation_function == ActivationFunctions.sigmoid:
            return ActivationFunctions.sigmoid_gradient


class ForwardProp:

    @staticmethod
    def compute_layer(input_data, weights, bias=None):
        # Need to ensure there are enough weights for number of features in input data
        assert (input_data.shape[1] == weights.shape[0])

        if bias is not None:
            # Need to ensure that there is a bias term for each hidden node
            assert (weights.shape[1] == bias.shape[1])

        return np.dot(input_data, weights) + bias if bias is not None else np.dot(input_data, weights)

    @staticmethod
    def forward_prop(num_layers, initial_input, layer_weights, layer_activation_functions=None, layer_biases=None):
        """
        :param layer_activation_functions: The activation functions to be used on each layer. Should be reference to the function at each key.
        :param layer_biases: the biases associated with every layer. Is of type dict
        :param num_layers: number of layers for the neural network
        :param initial_input: the input data
        :param layer_weights: the weights associated with every layer contained in a dictionary with the key being which layer they are for (starting at 1)
        :return: the ouput vector after the forward propogation
        """

        # This is done to ensure I can use the .get function later to return None
        if layer_biases is None:
            layer_biases = dict()
        else:
            assert (len(layer_biases) == num_layers)

        if layer_activation_functions is None:
            layer_activation_functions = dict()
        else:
            assert (len(layer_activation_functions) == num_layers)

        assert (isinstance(layer_weights, dict))
        assert (len(layer_weights) == num_layers)

        # Dictionary to keep track of inputs for each layer
        layer_input_dict = dict()

        current_input = initial_input
        for current_layer_number in range(1, num_layers + 1):
            # Get weights for current layer
            current_weights = layer_weights[current_layer_number]

            # Get bias vector for current layer. If there is no bias for that layer returns none
            current_bias = layer_biases.get(current_layer_number, None)

            # Get current activation function for the layer
            current_activation_function = layer_activation_functions.get(current_layer_number, None)

            # Get output matrix for current_layer
            output = ForwardProp.compute_layer(current_input, current_weights, current_bias)

            # If there is an activation function for the layer
            if current_activation_function:
                output = current_activation_function(output)

            layer_input_dict[current_layer_number] = current_input

            # The input into the next layer becomes the output from the previous layer
            current_input = output

        return current_input, layer_input_dict


class BackProp:

    def computer_layer_gradients(self):
        pass

    @staticmethod
    def back_prop(num_layers, layer_inputs, layer_weights, layer_activation_functions, expected_y, predicted_y):
        """
        :param layer_activation_functions: activation functions for each layer
        :param num_layers: number of layers
        :param layer_inputs: the inputs calculated for each layer. The key 1 should return the initial data we put in.
        :param layer_weights: the weights used in each layer of the neural network
        :param expected_y: what the expected value of the output should be. I.e. the real data
        :param predicted_y: What the neural network put out
        :return: two dicts, one for the weights and one for the biases which contains the gradients for each layer
        """

        assert (isinstance(layer_inputs, dict))
        assert (isinstance(layer_weights, dict))
        assert (isinstance(layer_activation_functions, dict))

        # Ensure information is defined for every layer
        assert (len(layer_activation_functions) == num_layers)
        assert (len(layer_weights) == num_layers)
        assert (len(layer_inputs) == num_layers)

        weight_gradients = dict()
        bias_gradients = dict()

        # Key 1 in layer_inputs should contain the initial data input. So the shape 0 will return number of rows, hence
        # number of examples
        n_examples = layer_inputs[1].shape[0]

        # This assumes that we will always use a SIGMOID activation function for the last output
        dz_last = (predicted_y - expected_y) * (1 / n_examples)
        # num_layers because we want the inputs into the last layer
        dw_last = np.dot(layer_inputs[num_layers].T, dz_last)
        db_last = np.sum(dz_last)

        # Set gradients for final layer
        weight_gradients[num_layers] = dw_last
        bias_gradients[num_layers] = db_last

        current_dz = dz_last
        # Have to go backwards in layer numbers. Start at num_layer-1 because last layer is always the same code as
        # above
        for current_layer_number in range(num_layers - 1, 0, -1):
            # Get the activation gradient function for the current activation function
            current_activation_gradient_function = ActivationFunctions.get_activation_gradient(
                layer_activation_functions[current_layer_number])

            current_dz, current_dw, current_db = BackProp.compute_gradient(next_layer_dz=current_dz,
                                                                           next_layer_weights=layer_weights[
                                                                               current_layer_number + 1],
                                                                           activation_gradient_function=current_activation_gradient_function,
                                                                           current_layer_input=layer_inputs[
                                                                               current_layer_number],
                                                                           next_layer_inputs=layer_inputs[
                                                                               current_layer_number + 1])

            # Store information of gradients for each layer
            weight_gradients[current_layer_number] = current_dw
            bias_gradients[current_layer_number] = current_db

        return weight_gradients, bias_gradients

    @staticmethod
    def compute_gradient(next_layer_dz, next_layer_weights, activation_gradient_function, current_layer_input,
                         next_layer_inputs):
        # Calculate dZ for current layer (we use weights from the next layer hence the +1)
        dz_current = np.dot(next_layer_dz,
                            next_layer_weights.T) * activation_gradient_function(next_layer_inputs)

        current_dw = np.dot(current_layer_input.T, dz_current)
        current_db = np.sum(dz_current, axis=0)

        return dz_current, current_dw, current_db


class NeuralNetwork:

    def __init__(self, x_train, y_train, layer_sizes, activation_function_dict, learning_rate=0.0001,
                 num_epochs=1000, batch_size=64):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.weights_dict = dict()
        self.bias_dict = dict()
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
    desired_architecture = [6, 6]
    nn_architecture = create_architecture(num_features, desired_architecture)

    # Defines the activation functions used for each layer
    activations_dict = {1: ActivationFunctions.relu, 2: ActivationFunctions.relu, 3: ActivationFunctions.sigmoid}

    neural_network = NeuralNetwork(x_train=data_train, y_train=labels_train, layer_sizes=nn_architecture,
                                   activation_function_dict=activations_dict, learning_rate=0.1, num_epochs=1000)

    epochs, cost = neural_network.optimise(error_stop=0.09)

    plt.plot(epochs, cost)
    plt.show()


if __name__ == '__main__':
    main()
