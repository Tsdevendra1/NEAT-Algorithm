import unittest
import numpy as np
from main import ForwardProp, ActivationFunctions, BackProp


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
                'prediction'].tolist(),
            expected_output_nobias.tolist())

        # Bias test
        self.assertEqual(ForwardProp.forward_prop(num_layers=2, initial_input=input_array, layer_weights=weight_dict,
                                                  layer_biases=bias_dict)['prediction'].tolist(),
                         expected_output_bias.tolist())

        # Testing Activation function with bias
        self.assertEqual(ForwardProp.forward_prop(num_layers=2, initial_input=input_array, layer_weights=weight_dict,
                                                  layer_biases=bias_dict,
                                                  layer_activation_functions=activation_function_dict)[
                             'prediction'].tolist(),
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

        output = ForwardProp.forward_prop(num_layers=1, initial_input=input_matrix, layer_weights=weights_dict,
                                          layer_activation_functions=activation_function_dict)

        # Excluded bias gradients here
        weight_gradients, _ = BackProp.back_prop(num_layers=1, layer_inputs=output['layer_input_dict'],
                                                              layer_weights=weights_dict, layer_activation_functions=activation_function_dict,
                                                              expected_y=expected_y, predicted_y=output['prediction'])

        self.assertEqual(weight_gradients[1].astype(int).tolist(), expected_weight_gradients.tolist())



if __name__ == '__main__':
    unittest.main()
