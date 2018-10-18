import numpy as np
import unittest


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

        if layer_activation_functions is None:
            layer_activation_functions = dict()

        assert (initial_input, np.array)
        assert (isinstance(layer_weights, dict))

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

        return {'prediction': current_input, 'layer_input_dict': layer_input_dict}


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

        weight_gradients = dict()
        bias_gradients = dict()

        # Key 1 in layer_inputs should contain the initial data input. So the shape 0 will return number of rows, hence
        # number of examples
        n_examples = layer_inputs[1].shape[0]

        random = layer_inputs[num_layers].T
        # This assumes that we will always use a sigmoid activation function for the last output
        dz_last = (predicted_y - expected_y) * (1 / n_examples)
        # num_layers because we want the inputs into the last layer
        dw_last = np.dot(layer_inputs[num_layers].T, dz_last)
        db_last = np.sum(dz_last)

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
                                                                           activation_gradient_function=current_activation_gradient_function)

            # Store information of gradients for each layer
            weight_gradients[current_layer_number] = current_dw
            bias_gradients[current_layer_number] = current_db

        return weight_gradients, bias_gradients

    @staticmethod
    def compute_gradient(next_layer_dz, next_layer_weights, activation_gradient_function, current_layer_input):
        # Calculate dZ for current layer (we use weights from the next layer hence the +1)
        dz_current = np.dot(next_layer_dz,
                            next_layer_weights.T) * activation_gradient_function(current_layer_input)

        current_dw = np.dot(current_layer_input.T, dz_current)
        current_db = np.sum(dz_current, axis=0)

        return dz_current, current_dw, current_db


if __name__ == '__main__':
    print(ActivationFunctions.sigmoid(np.array([[1, -1], [-1, 1]])))
