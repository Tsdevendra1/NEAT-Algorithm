import numpy as np


class ActivationFunctions:

    @staticmethod
    def relu(input_matrix):
        output = np.maximum(input_matrix, 0, input_matrix)

        return output

    @staticmethod
    def sigmoid(x):
        # TODO: Create the version for the genome using -4.9
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
            # Broadcast so we can remove the required bias for genes
            broadcasted_bias = np.broadcast_to(bias, (input_data.shape[0], bias.shape[1]))

        return np.dot(input_data, weights) + broadcasted_bias if bias is not None else np.dot(input_data, weights)

    @staticmethod
    def ensure_no_activation_applied(output_without_activation, output_with_activation, constant_connections,
                                     current_layer, node_map):
        """
        Ensures the activation function isn't applied to the nodes which are dummy nodes
        :param output_without_activation: The output with the activation function applied
        :param output_with_activation: The output with the activation function applied
        :param constant_connections: The connections where there should be an activation function applied
        :param current_layer: The current layer we're calculating in
        :param node_map: A dictionary of which number each node is in their respective layer
        :return:
        """
        # Need to keep the values where the
        for connection in constant_connections[current_layer]:
            # Need to convert to their position in the layer. Minus one because of python indexing
            output_position_within_layer = node_map[connection.output_node] - 1
            # The output node position is the node which shouldn't have any activations applied. So we use all the
            # values from before the activation was applied
            output_with_activation[:, output_position_within_layer] = \
                output_without_activation[
                :, output_position_within_layer]

        return output_with_activation

    @staticmethod
    def genome_forward_prop(num_layers, initial_input, layer_weights, keep_constant_connections, node_map,
                            layer_activation_functions, layer_biases):
        """
        :param no_activations_matrix_per_layer: A dict containing an array for each layer which showcases which biases should not be applied
        :param node_map: A dict for each node which shows which number node they are in their respective layer
        :param keep_constant_connections: A list of connection for which the connection should remain constand and no activation function applied
        :param layer_activation_functions: The activation functions to be used on each layer. Should be reference to the function at each key.
        :param layer_biases: the biases associated with every layer. Is of type dict
        :param num_layers: number of layers for the neural network
        :param initial_input: the input data
        :param layer_weights: the weights associated with every layer contained in a dictionary with the key being which layer they are for (starting at 1)
        :return: the ouput vector after the forward propogation
        """

        # This is done to ensure I can use the .get function later to return None
        if layer_biases is None:
            layer_biases = {}
        else:
            assert (len(layer_biases) == num_layers)

        if layer_activation_functions is None:
            layer_activation_functions = {}
        else:
            assert (len(layer_activation_functions) == num_layers)

        assert (isinstance(layer_weights, dict))
        assert (len(layer_weights) == num_layers)

        # Dictionary to keep track of inputs for each layer
        layer_input_dict = {}

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
                saved_output = output
                output = current_activation_function(output)
                output = ForwardProp.ensure_no_activation_applied(output_without_activation=saved_output,
                                                                  output_with_activation=output, node_map=node_map,
                                                                  constant_connections=keep_constant_connections,
                                                                  current_layer=current_layer_number)

            layer_input_dict[current_layer_number] = current_input

            # The input into the next layer becomes the output from the previous layer
            current_input = output

        return current_input, layer_input_dict

    @staticmethod
    def forward_prop(num_layers, initial_input, layer_weights, layer_activation_functions=None, layer_biases=None,
                     return_number_before_last_activation=False):
        """
        :param return_number_before_last_activation: If you want the raw output number instead of the sigmoid applied to it
        :param layer_activation_functions: The activation functions to be used on each layer. Should be reference to the function at each key.
        :param layer_biases: the biases associated with every layer. Is of type dict
        :param num_layers: number of layers for the neural network
        :param initial_input: the input data
        :param layer_weights: the weights associated with every layer contained in a dictionary with the key being which layer they are for (starting at 1)
        :return: the ouput vector after the forward propogation
        """

        # This is done to ensure I can use the .get function later to return None
        if layer_biases is None:
            layer_biases = {}
        else:
            assert (len(layer_biases) == num_layers)

        if layer_activation_functions is None:
            layer_activation_functions = {}
        else:
            assert (len(layer_activation_functions) == num_layers)

        assert (isinstance(layer_weights, dict))
        assert (len(layer_weights) == num_layers)

        # Dictionary to keep track of inputs for each layer
        layer_input_dict = {}

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

            # If you want to return the output before the sigmoid is applied on the last layer
            if return_number_before_last_activation and current_layer_number == num_layers:
                return output

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

        weight_gradients = {}
        bias_gradients = {}

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
