import unittest
import numpy as np
from main import ForwardProp, ActivationFunctions, BackProp, NeuralNetwork, create_architecture, create_data


class TestForwardProp(unittest.TestCase):

    def setUp(self):
        pass

    def test_compute_layer_single(self):
        """
        Tests compute_layer for a single perceptron
        :return:
        """
        # Single perceptron
        input_array = np.array([[1, 2], [2, 3], [3, 4]])
        weights = np.array([[1], [2]])
        bias = np.array([[1]])
        expected_output_nobias = np.array([[5], [8], [11]])
        # This includes a bias test
        expected_output_bias = np.array([[6], [9], [12]])

        # No bias test
        self.assertEqual(ForwardProp.compute_layer(input_array, weights).tolist(), expected_output_nobias.tolist())
        # Bias test
        self.assertEqual(ForwardProp.compute_layer(input_array, weights, bias).tolist(),
                         expected_output_bias.tolist())

    def test_compute_layer_multiple(self):
        """
        Tests compute_layer for multiple hidden nodes for a single layer
        :return:
        """
        input_array = np.array([[1, 2], [2, 3], [3, 4]])
        weights = np.array([[1, 2], [2, 1]])
        bias = np.array([[1, 2]])
        expected_output_nobias = np.array([[5, 4], [8, 7], [11, 10]])
        # This includes a bias test
        expected_output_bias = np.array([[6, 6], [9, 9], [12, 12]])

        # No bias test
        self.assertEqual(ForwardProp.compute_layer(input_array, weights).tolist(), expected_output_nobias.tolist())
        # Bias test
        self.assertEqual(ForwardProp.compute_layer(input_array, weights, bias).tolist(), expected_output_bias.tolist())

    def test_forward_prop(self):
        """
        Test for a one hidden layer neural network to see what the forward propagation output is
        """
        input_array = np.array([[1, 2], [2, 3], [3, 4]])
        layer_1_weights = np.array([[1, 2], [2, 1]])
        layer_2_weights = np.array([[3], [4]])
        layer_1_bias = np.array([[1, 2]])
        layer_2_bias = np.array([[1]])
        expected_output_nobias = np.array([[31], [52], [73]])
        expected_output_bias = np.array([[43], [64], [85]])

        weight_dict = {1: layer_1_weights, 2: layer_2_weights}
        bias_dict = {1: layer_1_bias, 2: layer_2_bias}
        activation_function_dict = {1: ActivationFunctions.relu, 2: ActivationFunctions.relu}

        # No bias test
        self.assertEqual(
            ForwardProp.forward_prop(num_layers=2, initial_input=input_array, layer_weights=weight_dict)[
                0].tolist(),
            expected_output_nobias.tolist())

        # Bias test
        self.assertEqual(ForwardProp.forward_prop(num_layers=2, initial_input=input_array, layer_weights=weight_dict,
                                                  layer_biases=bias_dict)[0].tolist(),
                         expected_output_bias.tolist())

        # Testing Activation function with bias
        self.assertEqual(ForwardProp.forward_prop(num_layers=2, initial_input=input_array, layer_weights=weight_dict,
                                                  layer_biases=bias_dict,
                                                  layer_activation_functions=activation_function_dict)[
                             0].tolist(),
                         expected_output_bias.tolist())


class TestActivationFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def test_relu(self):
        input_matrix = np.array([[1, -1], [-1, 1]])
        expected_output = np.array([[1, 0], [0, 1]])

        self.assertEqual(ActivationFunctions.relu(input_matrix).tolist(), expected_output.tolist())

    def test_sigmoid(self):
        input_matrix = np.array([[1, -1], [-1, 1]])
        expected_output = np.array([[0.73, 0.27], [0.27, 0.73]])

        function_output = np.around(ActivationFunctions.sigmoid(input_matrix), 2)

        self.assertEqual(function_output.tolist(), expected_output.tolist())


class TestBackProp(unittest.TestCase):

    def setUp(self):
        pass

    def test_back_prop_one_layer(self):
        """
        Test a very simple one layer neural network to see if correct gradients are calculated
        """
        # Initial data
        input_matrix = np.array([[1, 2], [3, 4]])
        weights = np.array([[2], [3]])
        expected_y = np.array([[2], [2]])
        expected_weight_gradients = np.array([[-2], [-3]])

        # Dictionary with layer information
        weights_dict = {1: weights}
        activation_function_dict = {1: ActivationFunctions.sigmoid}

        prediction, layer_input_dict = ForwardProp.forward_prop(num_layers=1, initial_input=input_matrix,
                                                                layer_weights=weights_dict,
                                                                layer_activation_functions=activation_function_dict)

        # Excluded bias gradients here
        weight_gradients, _ = BackProp.back_prop(num_layers=1, layer_inputs=layer_input_dict,
                                                 layer_weights=weights_dict,
                                                 layer_activation_functions=activation_function_dict,
                                                 expected_y=expected_y, predicted_y=prediction)

        self.assertEqual(weight_gradients[1].astype(int).tolist(), expected_weight_gradients.tolist())

    def test_back_prop_two_layer(self):
        input_matrix = np.array([[1, 2], [3, 4]])
        weights_1 = np.array([[2, 2], [3, 3]])
        weights_2 = np.array([[2], [3]])
        expected_y = np.array([[38], [88]])
        expected_weight_gradients_2 = np.array([[26], [26]])
        expected_weight_gradients_1 = np.array([[8, 12], [12, 18]])

        # Dictionary with layer information
        weights_dict = {1: weights_1, 2: weights_2}
        activation_function_dict = {1: ActivationFunctions.relu, 2: ActivationFunctions.relu}

        prediction, layer_input_dict = ForwardProp.forward_prop(num_layers=2, initial_input=input_matrix,
                                                                layer_weights=weights_dict,
                                                                layer_activation_functions=activation_function_dict)

        # Excluded bias gradients here
        weight_gradients, _ = BackProp.back_prop(num_layers=2, layer_inputs=layer_input_dict,
                                                 layer_weights=weights_dict,
                                                 layer_activation_functions=activation_function_dict,
                                                 expected_y=expected_y, predicted_y=prediction)

        self.assertEqual(np.round(weight_gradients[2], 0).astype(int).tolist(), expected_weight_gradients_2.tolist())
        self.assertEqual(np.round(weight_gradients[1], 0).astype(int).tolist(), expected_weight_gradients_1.tolist())


class TestNeuralNetworkOneLayer(unittest.TestCase):

    def setUp(self):
        self.data_train, self.labels_train = create_data(n_generated=5000)

        self.num_features = self.data_train.shape[1]

        #  This means it will be a two layer neural network with one layer being hidden with 2 nodes
        self.desired_architecture = [6]
        nn_architecture = create_architecture(self.num_features, self.desired_architecture)

        # Defines the activation functions used for each layer
        activations_dict = {1: ActivationFunctions.sigmoid, 2: ActivationFunctions.sigmoid}

        self.neural_network = NeuralNetwork(x_train=self.data_train, y_train=self.labels_train,
                                            layer_sizes=nn_architecture,
                                            activation_function_dict=activations_dict, learning_rate=0.1)

    def test_initialise_parameters_shapes(self):
        """
        Instead of testing for the values specifically we just test to ensure that the parameters initialise with the
        correct shape
        """
        expected_shape_layer_1 = (self.num_features, self.desired_architecture[0])
        # Number of features from last layer and because it's the last layer should only be one column
        expected_shape_layer_2 = (self.desired_architecture[0], 1)

        self.assertEqual(self.neural_network.weights_dict[1].shape, expected_shape_layer_1)
        self.assertEqual(self.neural_network.weights_dict[2].shape, expected_shape_layer_2)

    def test_optimise(self):
        epochs, cost = self.neural_network.optimise()

        # When this was working 0.002 was the error
        expected_error_after_1000_epochs = 0.002
        self.assertEqual(round(cost[999], 3), expected_error_after_1000_epochs)


class TestNeuralNetworkMultiLayer(unittest.TestCase):
    """
    Similar to above test case but this is a 2 layer hidden neural network instead of just one
    """

    def setUp(self):
        # Test and Train data
        data_train, labels_train = create_data(n_generated=5000)

        num_features = data_train.shape[1]

        #  This means it will be a two layer neural network with one layer being hidden with 2 nodes
        desired_architecture = [5, 6]
        nn_architecture = create_architecture(num_features, desired_architecture)

        # Defines the activation functions used for each layer
        activations_dict = {1: ActivationFunctions.sigmoid, 2: ActivationFunctions.sigmoid,
                            3: ActivationFunctions.sigmoid}

        self.neural_network = NeuralNetwork(x_train=data_train, y_train=labels_train, layer_sizes=nn_architecture,
                                            activation_function_dict=activations_dict, learning_rate=0.1)

    def test_optimise(self):
        epochs, cost = self.neural_network.optimise()

        # When this was working 0.002 was the error
        expected_error_after_1000_epochs = 0.0
        self.assertEqual(round(cost[999], 3), expected_error_after_1000_epochs)


if __name__ == '__main__':
    unittest.main()
