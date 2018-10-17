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
    def forward_prop(num_layers, initial_input, layer_weights, layer_biases=None):
        """
        :param layer_biases: the biases associated with every layer. Is of type dict
        :param num_layers: number of layers for the neural network
        :param initial_input: the input data
        :param layer_weights: the weights associated with every layer contained in a dictionary with the key being which layer they are for (starting at 1)
        :return: the ouput vector after the forward propogation
        """

        # This is done to ensure I can use the .get function later to return None
        if layer_biases is None:
            layer_biases = dict()

        assert (initial_input, np.array)
        assert (isinstance(layer_weights, dict))

        current_input = initial_input
        for current_layer_number in range(1, num_layers + 1):
            # Get weights for current layer
            current_weights = layer_weights[current_layer_number]

            # Get bias vector for current layer. If there is no bias for that layer returns none
            current_bias = layer_biases.get(current_layer_number, None)

            # Get output matrix for current_layer
            output = ForwardProp.compute_layer(current_input, current_weights, current_bias)

            # The input into the next layer becomes the output from the previous layer
            current_input = output

        return current_input


class BackProp:

    def computer_layer_gradients(self):
        pass

    @staticmethod
    def calc_gradients(num_layers, layer_inputs, layer_weights, expected_y, predicted_y):
        """
        :param num_layers: number of layers
        :param layer_inputs: the inputs calculated for each layer. The key 1 should return the initial data we put in.
        :param layer_weights: the weights used in each layer of the neural network
        :param expected_y: what the expected value of the output should be. I.e. the real data
        :param predicted_y: What the neural network put out
        :return: two dicts, one for the weights and one for the biases which contains the gradients for each layer
        """

        assert (isinstance(layer_inputs, dict))
        assert (isinstance(layer_weights, dict))

        weight_gradients = dict()
        bias_gradients = dict()

        # Key 1 in layer_inputs should contain the initial data input. So the shape 0 will return number of rows, hence
        # number of examples
        n_examples = layer_inputs[1].shape[0]

        dZ_last = (predicted_y - expected_y) * (1 / n_examples)
        dW_last = np.dot(layer_inputs[num_layers - 1], dZ_last)
        dB_last = np.sum(dZ_last)

    # def test(self):
    #         prediction, a1 = self.predict(data_batch, optimise=True)
    #
    #         # Compute gradients first layer
    #         dZ2 = (prediction - labels_batch) * (1 / n_examples)
    #         dW2 = np.dot(a1.T, dZ2)
    #         dB2 = np.sum(dZ2)
    #
    #         # Compute gradients second layer
    #         dZ1 = np.dot(dZ2, self.parameters['w2'].T) * self.sigmoid_gradient(a1)
    #         dW1 = np.dot(data_batch.T, dZ1)
    #         dB1 = np.sum(dZ1, axis=0)
    #
    #         gradients_dict = {'w1': dW1, 'w2': dW2, 'b1': dB1, 'b2': dB2}
    #
    #     return gradients_dict, prediction


if __name__ == '__main__':
    print(ActivationFunctions.sigmoid(np.array([[1, -1], [-1, 1]])))
