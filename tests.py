import unittest
import numpy as np

from config import Config
from genome_neural_network import GenomeNeuralNetwork
from neural_network import ForwardProp, ActivationFunctions, BackProp, NeuralNetwork, create_architecture, create_data
from deconstruct_genome import DeconstructGenome
from genome import Genome
from gene import ConnectionGene, NodeGene
from reproduce import Reproduce
from stagnation import Stagnation


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
        epochs, cost = self.neural_network.optimise(print_epoch_cost=False)

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
        epochs, cost = self.neural_network.optimise(print_epoch_cost=False)

        # When this was working 0.002 was the error
        expected_error_after_1000_epochs = 0.0
        self.assertEqual(round(cost[999], 3), expected_error_after_1000_epochs)


class TestDeconstructGenomeClass(unittest.TestCase):

    def setUp(self):
        node_list = [NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='source'),
                     NodeGene(node_id=3, node_type='hidden'),
                     NodeGene(node_id=4, node_type='hidden'),
                     NodeGene(node_id=5, node_type='output')]

        connection_list = [ConnectionGene(input_node=1, output_node=3, innovation_number=1),
                           ConnectionGene(input_node=1, output_node=4, innovation_number=2),
                           ConnectionGene(input_node=2, output_node=3, innovation_number=3),
                           ConnectionGene(input_node=2, output_node=4, innovation_number=4),
                           ConnectionGene(input_node=3, output_node=5, innovation_number=5),
                           ConnectionGene(input_node=4, output_node=5, innovation_number=6)]

        self.genome = Genome(nodes=node_list, connections=connection_list, key=1)

    def test_get_node_layer(self):
        expected_answer = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3}
        self.assertEqual(
            DeconstructGenome.get_node_layers(connections=list(self.genome.connections.values()), genome=self.genome)[
                0],
            expected_answer)

    def test_unpack_genome(self):
        expected_answer = np.ones((2, 2))

        # check that the first layer weights are the correct ones
        self.assertEqual(self.genome.connection_matrices_per_layer[1].tolist(), expected_answer.tolist())

    def test_unpack_genome_broken_link(self):
        """
        Tests unpack genome for a genome which contains a broken link
        """
        expected_answer = np.ones((2, 2))
        expected_answer[0, 0] = 0
        node_list = [NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='source'),
                     NodeGene(node_id=3, node_type='hidden'),
                     NodeGene(node_id=4, node_type='hidden'),
                     NodeGene(node_id=5, node_type='output')]

        # Note that one of the connections isn't enabled
        connection_list = [ConnectionGene(input_node=1, output_node=3, innovation_number=1, enabled=False),
                           ConnectionGene(input_node=1, output_node=4, innovation_number=2, enabled=True),
                           ConnectionGene(input_node=2, output_node=3, innovation_number=3, enabled=True),
                           ConnectionGene(input_node=2, output_node=4, innovation_number=4, enabled=True),
                           ConnectionGene(input_node=3, output_node=5, innovation_number=5, enabled=True),
                           ConnectionGene(input_node=4, output_node=5, innovation_number=6, enabled=True)]

        genome = Genome(connections=connection_list, nodes=node_list, key=2)

        self.assertEqual(genome.connection_matrices_per_layer[1].tolist(), expected_answer.tolist())

    def test_forward_prop(self):
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

        x_data = np.array([[1, 2]])
        y_data = np.array([[1]])
        genome_nn = GenomeNeuralNetwork(genome=genome, x_train=x_data, y_train=y_data, learning_rate=0.1,
                                        create_weights_bias_from_genome=True, activation_type='relu')

        expected_answer = np.array([[104.5]])

        # This works because the activation type is relu and there aren't any negative numbers. But it should ideally be
        # genome_forward_prop instead of forward_prop
        output = ForwardProp.forward_prop(num_layers=genome_nn.num_layers, initial_input=x_data,
                                          layer_weights=genome_nn.weights_dict,
                                          layer_activation_functions=genome_nn.activation_function_dict,
                                          layer_biases=genome_nn.bias_dict, return_number_before_last_activation=True)

        self.assertEqual(expected_answer.tolist(), output.tolist())


class TestGenomeUnpack(unittest.TestCase):
    def setUp(self):
        pass

    def test_unpack_genome_3(self):
        """
        Testing another genome which would normally fail if the unpack genome method is not coded correctly
        """
        for i in range(1000):
            node_list = [NodeGene(node_id=1, node_type='source'),
                         NodeGene(node_id=2, node_type='source'),
                         NodeGene(node_id=3, node_type='output')]

            connection_list = [ConnectionGene(input_node=1, output_node=3, innovation_number=1),
                               ConnectionGene(input_node=2, output_node=3, innovation_number=2)]

            genome = Genome(connections=connection_list, nodes=node_list, key=1)
            self.assertTrue(genome)

        node_list = [NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=3, node_type='output')]

        connection_list = [ConnectionGene(input_node=1, output_node=3, innovation_number=1)]

        genome_2 = Genome(connections=connection_list, nodes=node_list, key=2)
        self.assertTrue(genome_2)

    def test_unpack_genome_2(self):
        """
        Testing another genome which would normally fail if the unpack genome method is not coded correctly
        """
        node_list = [NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='source'),
                     NodeGene(node_id=3, node_type='output'),
                     NodeGene(node_id=4, node_type='hidden')]

        connection_list = [ConnectionGene(input_node=1, output_node=4, innovation_number=1),
                           ConnectionGene(input_node=2, output_node=3, innovation_number=2),
                           ConnectionGene(input_node=4, output_node=3, innovation_number=6)]

        genome = Genome(connections=connection_list, nodes=node_list, key=1)
        self.assertTrue(genome)

    def test_unpack_genome_4(self):
        node_list = [NodeGene(node_id=0, node_type='source'),
                     NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=3, node_type='hidden', bias=0.5),
                     NodeGene(node_id=4, node_type='hidden', bias=-1.5),
                     NodeGene(node_id=5, node_type='hidden', bias=0.5),
                     NodeGene(node_id=6, node_type='hidden', bias=-1.5),
                     NodeGene(node_id=2, node_type='output', bias=1.5)]

        connection_list = [ConnectionGene(input_node=1, output_node=4, innovation_number=1, enabled=True, weight=9),
                           ConnectionGene(input_node=4, output_node=2, innovation_number=5, enabled=True, weight=3),
                           ConnectionGene(input_node=0, output_node=5, innovation_number=2, enabled=True, weight=2),
                           ConnectionGene(input_node=5, output_node=3, innovation_number=4, enabled=True, weight=4),
                           ConnectionGene(input_node=3, output_node=4, innovation_number=3, enabled=True, weight=3),
                           ConnectionGene(input_node=1, output_node=2, innovation_number=7, enabled=True, weight=7)]

        genome = Genome(connections=connection_list, nodes=node_list, key=1)

        self.assertTrue(genome)

    def test_unpack_genome_5(self):
        # Multiple times due to randomisation of various elements
        for i in range(100):
            node_list = [NodeGene(node_id=0, node_type='source'),
                         NodeGene(node_id=1, node_type='source'),
                         NodeGene(node_id=2, node_type='output', bias=1.5),
                         NodeGene(node_id=3, node_type='hidden', bias=0.5),
                         NodeGene(node_id=4, node_type='hidden', bias=-1.5),
                         NodeGene(node_id=5, node_type='hidden', bias=0.5)]

            connection_list = [ConnectionGene(input_node=0, output_node=2, innovation_number=1, enabled=True, weight=9),
                               ConnectionGene(input_node=1, output_node=5, innovation_number=5, enabled=True, weight=3),
                               ConnectionGene(input_node=3, output_node=4, innovation_number=2, enabled=True, weight=2),
                               ConnectionGene(input_node=5, output_node=3, innovation_number=4, enabled=True, weight=4),
                               ConnectionGene(input_node=4, output_node=2, innovation_number=3, enabled=True, weight=3)]

            genome = Genome(connections=connection_list, nodes=node_list, key=1)

            self.assertTrue(genome)

    def test_unpack_genome_6(self):
        # Multiple times due to randomisation of various elements
        for i in range(100):
            node_list = [NodeGene(node_id=0, node_type='source'),
                         NodeGene(node_id=1, node_type='source'),
                         NodeGene(node_id=2, node_type='output', bias=1.5),
                         NodeGene(node_id=3, node_type='hidden', bias=0.5),
                         NodeGene(node_id=4, node_type='hidden', bias=-1.5),
                         NodeGene(node_id=5, node_type='hidden', bias=0.5)]

            connection_list = [ConnectionGene(input_node=0, output_node=2, innovation_number=1, enabled=True, weight=9),
                               ConnectionGene(input_node=1, output_node=2, innovation_number=5, enabled=True, weight=3),
                               ConnectionGene(input_node=0, output_node=5, innovation_number=2, enabled=True, weight=2),
                               ConnectionGene(input_node=3, output_node=4, innovation_number=4, enabled=True, weight=4),
                               ConnectionGene(input_node=5, output_node=3, innovation_number=3, enabled=True, weight=3),
                               ConnectionGene(input_node=4, output_node=2, innovation_number=9, enabled=True, weight=3)]

            genome = Genome(connections=connection_list, nodes=node_list, key=1)

            self.assertTrue(genome)

    def test_unpack_genome_7(self):
        # Multiple times due to randomisation of various elements
        node_list = [NodeGene(node_id=0, node_type='source'),
                     NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='output', bias=1.5),
                     NodeGene(node_id=3, node_type='hidden', bias=0.5),
                     NodeGene(node_id=4, node_type='hidden', bias=-1.5),
                     NodeGene(node_id=5, node_type='hidden', bias=-1.5),
                     NodeGene(node_id=6, node_type='hidden', bias=-1.5),
                     NodeGene(node_id=7, node_type='hidden', bias=0.5)]

        # TODO: the problem is that there is no input node with a 6
        # TODO:  if it doens't belong to a path like the ones with 6's turn it off
        connection_list = [ConnectionGene(input_node=0, output_node=2, innovation_number=1, enabled=False, weight=9),
                           ConnectionGene(input_node=3, output_node=5, innovation_number=2, enabled=True, weight=3),
                           ConnectionGene(input_node=5, output_node=2, innovation_number=3, enabled=True, weight=2),
                           ConnectionGene(input_node=1, output_node=5, innovation_number=4, enabled=True, weight=4),
                           ConnectionGene(input_node=1, output_node=2, innovation_number=5, enabled=False, weight=3),
                           ConnectionGene(input_node=5, output_node=6, innovation_number=6, enabled=True, weight=3),
                           ConnectionGene(input_node=3, output_node=6, innovation_number=7, enabled=True, weight=3),
                           ConnectionGene(input_node=1, output_node=7, innovation_number=8, enabled=True, weight=3),
                           ConnectionGene(input_node=7, output_node=2, innovation_number=9, enabled=True, weight=3),
                           ConnectionGene(input_node=3, output_node=2, innovation_number=10, enabled=True, weight=3),
                           ConnectionGene(input_node=1, output_node=3, innovation_number=11, enabled=True, weight=3)]

        genome = Genome(connections=connection_list, nodes=node_list, key=1)

        self.assertTrue(genome.connections[6].enabled is False)
        self.assertTrue(genome.connections[7].enabled is False)

        self.assertTrue(genome)

    def test_genome_unpack_8(self):
        node_list = [NodeGene(node_id=0, node_type='source'),
                     NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='output', bias=1.5),
                     NodeGene(node_id=3, node_type='hidden', bias=0.5),
                     NodeGene(node_id=4, node_type='hidden', bias=-1.5),
                     NodeGene(node_id=5, node_type='hidden', bias=-1.5),
                     NodeGene(node_id=6, node_type='hidden', bias=-1.5),
                     NodeGene(node_id=7, node_type='hidden', bias=0.5)]

        connection_list = [ConnectionGene(input_node=0, output_node=2, innovation_number=1, enabled=False, weight=9),
                           ConnectionGene(input_node=0, output_node=4, innovation_number=2, enabled=True, weight=3),
                           ConnectionGene(input_node=3, output_node=5, innovation_number=3, enabled=True, weight=2),
                           ConnectionGene(input_node=5, output_node=4, innovation_number=4, enabled=True, weight=4),
                           ConnectionGene(input_node=1, output_node=6, innovation_number=5, enabled=True, weight=3),
                           ConnectionGene(input_node=6, output_node=2, innovation_number=6, enabled=True, weight=3),
                           ConnectionGene(input_node=1, output_node=2, innovation_number=7, enabled=False, weight=3),
                           ConnectionGene(input_node=1, output_node=7, innovation_number=8, enabled=True, weight=3),
                           ConnectionGene(input_node=7, output_node=2, innovation_number=9, enabled=True, weight=3),
                           ConnectionGene(input_node=0, output_node=3, innovation_number=10, enabled=True, weight=3),
                           ConnectionGene(input_node=3, output_node=2, innovation_number=11, enabled=False, weight=3),
                           ConnectionGene(input_node=3, output_node=4, innovation_number=12, enabled=False, weight=3),
                           ConnectionGene(input_node=4, output_node=2, innovation_number=13, enabled=True, weight=3)]

        genome = Genome(connections=connection_list, nodes=node_list, key=1)

        self.assertTrue(len(genome.connections) == len(connection_list))
        self.assertTrue(list(genome.connections.values()) == connection_list)

        self.assertTrue(genome)


class TestGenomeNeuralNetwork(unittest.TestCase):
    def setUp(self):
        pass

    def test_creation_genome_nn(self):
        node_list = [NodeGene(node_id=0, node_type='source'),
                     NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='output', bias=0.5),
                     NodeGene(node_id=3, node_type='hidden', bias=1.2)]

        connection_list = [ConnectionGene(input_node=0, output_node=2, innovation_number=1, enabled=True, weight=9),
                           ConnectionGene(input_node=1, output_node=2, innovation_number=6, enabled=False, weight=5),
                           ConnectionGene(input_node=1, output_node=3, innovation_number=2, enabled=False, weight=5),
                           ConnectionGene(input_node=3, output_node=2, innovation_number=7, enabled=True, weight=7)]

        genome = Genome(connections=connection_list, nodes=node_list, key=1)
        x_data = np.array([[1, 0]])
        y_data = np.array([[1]])
        genome_nn = GenomeNeuralNetwork(genome=genome, create_weights_bias_from_genome=False, activation_type='sigmoid',
                                        x_train=x_data, y_train=y_data)

    def test_update_gene(self):
        node_list = [NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='source'),
                     NodeGene(node_id=3, node_type='hidden', bias=0.5),
                     NodeGene(node_id=4, node_type='hidden', bias=-1.5),
                     NodeGene(node_id=5, node_type='output', bias=1.5)]

        connection_list = [ConnectionGene(input_node=1, output_node=5, innovation_number=1, enabled=True, weight=9),
                           ConnectionGene(input_node=1, output_node=3, innovation_number=2, enabled=True, weight=2),
                           ConnectionGene(input_node=2, output_node=3, innovation_number=3, enabled=True, weight=3),
                           ConnectionGene(input_node=2, output_node=4, innovation_number=4, enabled=True, weight=4),
                           ConnectionGene(input_node=3, output_node=5, innovation_number=6, enabled=True, weight=5),
                           ConnectionGene(input_node=4, output_node=5, innovation_number=7, enabled=True, weight=7)]

        genome = Genome(connections=connection_list, nodes=node_list, key=1)
        x_data = np.array([[1, 0]])
        y_data = np.array([[1]])
        genome_nn = GenomeNeuralNetwork(genome=genome, create_weights_bias_from_genome=False, activation_type='sigmoid',
                                        x_train=x_data, y_train=y_data)

        genome_nn.weights_dict[1] = np.array([[3, 0, 1], [4, 5, 0]])
        genome_nn.weights_dict[2] = np.array([[1], [2], [3]])

        genome_nn.bias_dict[1] = np.array([[1, 2, 0]])
        genome_nn.bias_dict[2] = np.array([[7]])

        expect_weights = {(1, 5): 3, (1, 3): 3, (2, 3): 4, (2, 4): 5, (4, 5): 2, (3, 5): 1}

        expected_bias = {3: 1, 4: 2, 5: 7}

        genome_nn.update_genes()

        for connection in genome.connections.values():
            self.assertEqual(expect_weights[(connection.input_node, connection.output_node)], connection.weight)

        for node in genome.nodes.values():
            if node.node_type != 'source':
                self.assertEqual(node.bias, expected_bias[node.node_id])

    def test_run_one_pass(self):
        node_list = [NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='source'),
                     NodeGene(node_id=5, node_type='output', bias=0)]

        connection_list = [ConnectionGene(input_node=1, output_node=5, innovation_number=1, enabled=True, weight=4),
                           ConnectionGene(input_node=2, output_node=5, innovation_number=7, enabled=True, weight=0)]

        genome = Genome(nodes=node_list, connections=connection_list, key=1)

        x_data = np.array([[1, 0]])
        y_data = np.array([[1]])
        genome_nn = GenomeNeuralNetwork(genome=genome, x_train=x_data, y_train=y_data, learning_rate=0.1,
                                        create_weights_bias_from_genome=True, activation_type='sigmoid')

        cost = genome_nn.run_one_pass(input_data=x_data, labels=y_data)
        # The output should be sigmoid of the prediction which is 4. Then the loss is calculated using log loss.
        expected_answer = round(-np.log(ActivationFunctions.sigmoid(4)), 5)

        self.assertEqual(expected_answer, round(cost, 5))


class TestGenomeMutatation(unittest.TestCase):

    def setUp(self):
        node_list = [NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='source'),
                     NodeGene(node_id=3, node_type='hidden'),
                     NodeGene(node_id=4, node_type='hidden'),
                     NodeGene(node_id=5, node_type='output')]

        # Note that one of the connections isn't enabled
        connection_list = [ConnectionGene(input_node=1, output_node=3, innovation_number=1, enabled=True),
                           ConnectionGene(input_node=1, output_node=4, innovation_number=2, enabled=True),
                           ConnectionGene(input_node=2, output_node=3, innovation_number=3, enabled=True),
                           ConnectionGene(input_node=2, output_node=4, innovation_number=4, enabled=True),
                           ConnectionGene(input_node=3, output_node=5, innovation_number=5, enabled=True),
                           ConnectionGene(input_node=4, output_node=5, innovation_number=6, enabled=True)]

        self.genome = Genome(connections=connection_list, nodes=node_list, key=2)

    def test_add_connection_2(self):
        """
        Can't let two source nodes connect to each other
        :return:
        """
        node_list = [NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='source'),
                     NodeGene(node_id=3, node_type='output', bias=0)]

        connection_list = [ConnectionGene(input_node=1, output_node=3, innovation_number=1),
                           ConnectionGene(input_node=2, output_node=3, innovation_number=2)]

        for i in range(100):
            genome = Genome(connections=connection_list, nodes=node_list, key=1)
            reproduce = Reproduce(config=Config, stagnation=Stagnation)
            reproduce.global_innovation_number = 7
            genome.add_connection(reproduction_instance=reproduce, innovation_tracker={})
            self.assertTrue(len(genome.connections) == 2)

    def test_add_connection(self):
        reproduce = Reproduce(config=Config, stagnation=Stagnation)
        reproduce.global_innovation_number = 7
        new_connection = self.genome.add_connection(
            reproduction_instance=reproduce,
            innovation_tracker={})

        self.assertTrue(len(self.genome.connections) == 7)

        # Unpack the new genome
        self.genome.unpack_genome()

        # Check if connection was where it connected to a node on the same layer
        if (new_connection.input_node == 3 and new_connection.output_node == 4) or (
                new_connection.input_node == 4 and new_connection.output_node == 3):
            # If the connection is one the same layer then the number of layers will have increased
            self.assertEqual(self.genome.num_layers_including_input, 4)

        # Check that two source nodes aren't connected
        elif self.genome.nodes[new_connection.input_node].node_type == 'source':
            self.assertTrue(self.genome.nodes[new_connection.output_node].node_type != 'source')

        # Can't connect to itself
        self.assertTrue(new_connection.input_node != new_connection.output_node)

    def test_remove_connection(self):
        node_list = [NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='source'),
                     NodeGene(node_id=3, node_type='hidden'),
                     NodeGene(node_id=4, node_type='hidden'),
                     NodeGene(node_id=5, node_type='output')]

        # Note that one of the connections isn't enabled
        connection_list = [
            ConnectionGene(input_node=2, output_node=4, innovation_number=4, enabled=True),
            ConnectionGene(input_node=4, output_node=3, innovation_number=5, enabled=True),
            ConnectionGene(input_node=3, output_node=5, innovation_number=8, enabled=True)]

        genome = Genome(connections=connection_list, nodes=node_list, key=2)
        genome.remove_connection()
        # TODO: Check how this works
        # Check that the source nodes are always there
        self.assertTrue(genome.nodes[1])
        self.assertTrue(genome.nodes[2])
        self.assertTrue(genome.nodes[5])

        genome.unpack_genome()

    def test_remove_connections_2(self):
        for i in range(100):
            node_list = [NodeGene(node_id=1, node_type='source'),
                         NodeGene(node_id=2, node_type='source'),
                         NodeGene(node_id=3, node_type='output', bias=0)]

            connection_list = [ConnectionGene(input_node=1, output_node=3, innovation_number=1),
                               ConnectionGene(input_node=2, output_node=3, innovation_number=2)]

            genome = Genome(connections=connection_list, nodes=node_list, key=2)
            genome.remove_connection()
            genome.unpack_genome()
            self.assertTrue(genome)

    def test_add_node(self):
        number_of_beginning_connections = len(self.genome.connections)
        self.assertEqual(number_of_beginning_connections, 6)

        # Because we replaced 1 connection with two
        expected_number_connections = number_of_beginning_connections + 2

        reproduce = Reproduce(config=Config, stagnation=Stagnation)
        reproduce.global_innovation_number = 7
        self.genome.add_node(reproduction_instance=reproduce,
                             innovation_tracker={})

        self.assertEqual(len(self.genome.connections), expected_number_connections)

        number_of_disabled_connections = 1

        disabled_counters = 0
        for connection in self.genome.connections.values():
            if not connection.enabled:
                disabled_counters += 1

        # One connection should have been disabled from the node addition
        self.assertEqual(number_of_disabled_connections, disabled_counters)

        self.assertTrue(self.genome.connections[8].weight == 1)

    def test_remove_node(self):
        node_list = [NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='source'),
                     NodeGene(node_id=3, node_type='hidden'),
                     NodeGene(node_id=4, node_type='hidden'),
                     NodeGene(node_id=5, node_type='output')]

        connection_list = [ConnectionGene(input_node=1, output_node=3, innovation_number=1, enabled=True, weight=1),
                           ConnectionGene(input_node=1, output_node=4, innovation_number=2, enabled=True, weight=2),
                           ConnectionGene(input_node=2, output_node=3, innovation_number=3, enabled=True, weight=3),
                           ConnectionGene(input_node=2, output_node=4, innovation_number=4, enabled=True, weight=4),
                           ConnectionGene(input_node=3, output_node=5, innovation_number=5, enabled=True, weight=5),
                           ConnectionGene(input_node=4, output_node=5, innovation_number=6, enabled=True, weight=6)]

        genome = Genome(connections=connection_list, nodes=node_list, key=1)

        genome.remove_node()
        genome.unpack_genome()

    def test_cross_over(self):
        node_list_1 = [NodeGene(node_id=1, node_type='source'),
                       NodeGene(node_id=2, node_type='source'),
                       NodeGene(node_id=3, node_type='hidden'),
                       NodeGene(node_id=4, node_type='hidden'),
                       NodeGene(node_id=5, node_type='output')]

        connection_list_1 = [ConnectionGene(input_node=1, output_node=3, innovation_number=1, enabled=True, weight=1),
                             ConnectionGene(input_node=1, output_node=4, innovation_number=2, enabled=True, weight=2),
                             ConnectionGene(input_node=2, output_node=3, innovation_number=3, enabled=True, weight=3),
                             ConnectionGene(input_node=2, output_node=4, innovation_number=4, enabled=True, weight=4),
                             ConnectionGene(input_node=3, output_node=5, innovation_number=5, enabled=True, weight=5),
                             ConnectionGene(input_node=4, output_node=5, innovation_number=6, enabled=True, weight=6)]

        node_list_2 = [NodeGene(node_id=1, node_type='source'),
                       NodeGene(node_id=2, node_type='source'),
                       NodeGene(node_id=3, node_type='hidden'),
                       NodeGene(node_id=4, node_type='hidden'),
                       NodeGene(node_id=5, node_type='output')]

        connection_list_2 = [ConnectionGene(input_node=1, output_node=3, innovation_number=1, enabled=True, weight=2),
                             ConnectionGene(input_node=2, output_node=3, innovation_number=3, enabled=True, weight=5),
                             ConnectionGene(input_node=2, output_node=4, innovation_number=4, enabled=True, weight=9),
                             ConnectionGene(input_node=1, output_node=5, innovation_number=7, enabled=True, weight=7),
                             ConnectionGene(input_node=2, output_node=5, innovation_number=8, enabled=True, weight=7),
                             ConnectionGene(input_node=3, output_node=5, innovation_number=5, enabled=True, weight=1),
                             ConnectionGene(input_node=4, output_node=5, innovation_number=6, enabled=True, weight=8)]

        genome_1 = Genome(connections=connection_list_1, nodes=node_list_1, key=1)
        genome_1.fitness = 3
        genome_2 = Genome(connections=connection_list_2, nodes=node_list_2, key=2)
        genome_2.fitness = 1

        child = Genome(key=4)
        child.crossover(genome_1=genome_1, genome_2=genome_2, config=Config)

        expected_genes = [1, 2, 3, 4, 5, 6]
        actual_genes = []
        for connection in child.connections.values():
            actual_genes.append(connection.innovation_number)

        self.assertEqual(expected_genes, actual_genes)

    def test_compatibility_distance(self):
        node_list_1 = [NodeGene(node_id=1, node_type='source'),
                       NodeGene(node_id=2, node_type='source'),
                       NodeGene(node_id=3, node_type='hidden'),
                       NodeGene(node_id=4, node_type='hidden'),
                       NodeGene(node_id=5, node_type='output')]

        connection_list_1 = [ConnectionGene(input_node=1, output_node=3, innovation_number=1, enabled=True, weight=1),
                             ConnectionGene(input_node=1, output_node=4, innovation_number=2, enabled=True, weight=2),
                             ConnectionGene(input_node=2, output_node=3, innovation_number=3, enabled=True, weight=3),
                             ConnectionGene(input_node=2, output_node=4, innovation_number=4, enabled=True, weight=4),
                             ConnectionGene(input_node=3, output_node=5, innovation_number=5, enabled=True, weight=5),
                             ConnectionGene(input_node=4, output_node=5, innovation_number=8, enabled=True, weight=6)]

        connection_list_2 = [ConnectionGene(input_node=1, output_node=3, innovation_number=1, enabled=True, weight=1),
                             ConnectionGene(input_node=1, output_node=4, innovation_number=2, enabled=True, weight=2),
                             ConnectionGene(input_node=2, output_node=3, innovation_number=3, enabled=True, weight=3),
                             ConnectionGene(input_node=2, output_node=4, innovation_number=4, enabled=True, weight=4),
                             ConnectionGene(input_node=3, output_node=5, innovation_number=5, enabled=True, weight=5),
                             ConnectionGene(input_node=4, output_node=5, innovation_number=6, enabled=True, weight=6),
                             ConnectionGene(input_node=4, output_node=5, innovation_number=7, enabled=True, weight=6),
                             ConnectionGene(input_node=4, output_node=5, innovation_number=9, enabled=True, weight=6),
                             ConnectionGene(input_node=4, output_node=5, innovation_number=10, enabled=True, weight=6)]

        genome_1 = Genome(connections=connection_list_1, nodes=node_list_1, key=1)
        genome_2 = Genome(connections=connection_list_1, nodes=node_list_1, key=2)
        genome_3 = Genome(connections=connection_list_2, nodes=node_list_1, key=3)

        compatibility_distance_1 = genome_1.compute_compatibility_distance(other_genome=genome_2, config=Config)
        compatibility_distance_2 = genome_2.compute_compatibility_distance(other_genome=genome_2, config=Config)

        compatibility_distance_3 = genome_3.compute_compatibility_distance(other_genome=genome_1, config=Config)
        compatibility_distance_4 = genome_1.compute_compatibility_distance(other_genome=genome_3, config=Config)

        self.assertTrue(compatibility_distance_1 == 0)
        self.assertTrue(compatibility_distance_1 == compatibility_distance_2)
        self.assertEqual(compatibility_distance_3, 5)
        self.assertEqual(compatibility_distance_3, compatibility_distance_4)

    def test_compatibility_distance_2(self):
        node_list_1 = [NodeGene(node_id=0, node_type='source'),
                       NodeGene(node_id=1, node_type='source'),
                       NodeGene(node_id=2, node_type='output'),
                       NodeGene(node_id=4, node_type='hidden'),
                       NodeGene(node_id=5, node_type='hidden')]

        connection_list_1 = [ConnectionGene(input_node=0, output_node=2, innovation_number=1, enabled=True, weight=1),
                             ConnectionGene(input_node=1, output_node=4, innovation_number=6, enabled=True, weight=2),
                             ConnectionGene(input_node=4, output_node=2, innovation_number=8, enabled=True, weight=3),
                             ConnectionGene(input_node=4, output_node=5, innovation_number=10, enabled=True, weight=4),
                             ConnectionGene(input_node=5, output_node=2, innovation_number=11, enabled=True, weight=5),
                             ConnectionGene(input_node=0, output_node=4, innovation_number=12, enabled=True, weight=6)]

        node_list_2 = [NodeGene(node_id=0, node_type='source'),
                       NodeGene(node_id=1, node_type='source'),
                       NodeGene(node_id=2, node_type='output'),
                       NodeGene(node_id=3, node_type='hidden')]

        connection_list_2 = [ConnectionGene(input_node=1, output_node=2, innovation_number=2, enabled=True, weight=1),
                             ConnectionGene(input_node=1, output_node=3, innovation_number=3, enabled=True, weight=2),
                             ConnectionGene(input_node=3, output_node=2, innovation_number=4, enabled=True, weight=3)]

        genome_1 = Genome(connections=connection_list_1, nodes=node_list_1, key=1)
        genome_2 = Genome(connections=connection_list_2, nodes=node_list_2, key=2)

        genome_2.fitness = -0.974

        compatibility_distance_1 = genome_1.compute_compatibility_distance(other_genome=genome_2, config=Config)
        compatibility_distance_2 = genome_2.compute_compatibility_distance(other_genome=genome_1, config=Config)

        # The problem is that genome_1 has no fitness
        self.assertTrue(compatibility_distance_2 == compatibility_distance_1)


class TestNEATClass(unittest.TestCase):

    def setUp(self):
        pass

    def test_genome_forward_pass_after_connection_removed(self):
        """
        Test that when there isn't a connection to the source, we delete that column in the x_data
        :return:
        """
        node_list = [NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='source'),
                     NodeGene(node_id=3, node_type='output', bias=0)]

        connection_list = [ConnectionGene(input_node=1, output_node=3, innovation_number=1, weight=np.random.randn())]

        x_data, y_data = create_data(n_generated=5000)

        genome = Genome(connections=connection_list, nodes=node_list, key=2)

        genome_nn = GenomeNeuralNetwork(genome=genome, x_train=x_data, y_train=y_data, learning_rate=0.1,
                                        create_weights_bias_from_genome=True, activation_type='relu')

        # Because we deleted one column due to there not being a connection to the second source
        self.assertTrue(genome_nn.x_train.shape[1] == 1)

        cost = genome_nn.run_one_pass(input_data=genome_nn.x_train, labels=y_data, return_cost_only=True)

        self.assertTrue(cost)


class TestConnectionDisabled(unittest.TestCase):
    def setUp(self):
        pass

    def test_connection_disabled_along_path(self):
        node_list = [NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='source'),
                     NodeGene(node_id=4, node_type='hidden'),
                     NodeGene(node_id=5, node_type='hidden'),
                     NodeGene(node_id=3, node_type='output', bias=0)]

        connection_list = [
            ConnectionGene(input_node=1, output_node=5, innovation_number=1, weight=np.random.randn(), enabled=False),
            ConnectionGene(input_node=2, output_node=4, innovation_number=4, weight=np.random.randn(), enabled=True),
            ConnectionGene(input_node=5, output_node=4, innovation_number=3, weight=np.random.randn(), enabled=True),
            ConnectionGene(input_node=4, output_node=3, innovation_number=2, weight=np.random.randn(), enabled=True)]
        self.assertTrue(connection_list[2].enabled is True)
        genome = Genome(connections=connection_list, nodes=node_list, key=1)

        # See check_any_disabled_connections_in_path which should set it to false since one of the connections in it's
        # path is disabled
        self.assertTrue(connection_list[2].enabled is False)

    def test_check_num_paths(self):
        node_list = [NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='source'),
                     NodeGene(node_id=4, node_type='hidden'),
                     NodeGene(node_id=5, node_type='hidden'),
                     NodeGene(node_id=3, node_type='output', bias=0)]

        connection_list = [
            ConnectionGene(input_node=1, output_node=5, innovation_number=1, weight=np.random.randn(), enabled=False),
            ConnectionGene(input_node=2, output_node=4, innovation_number=4, weight=np.random.randn(), enabled=False),
            ConnectionGene(input_node=5, output_node=4, innovation_number=3, weight=np.random.randn(), enabled=False),
            ConnectionGene(input_node=4, output_node=3, innovation_number=2, weight=np.random.randn(), enabled=True)]

        genome = Genome(key=1)
        genome.configure_genes(connections=connection_list, nodes=node_list)
        self.assertTrue(genome.unpack_genome() is False)

    def test_layer_nodes(self):
        node_list = [NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='source'),
                     NodeGene(node_id=4, node_type='hidden'),
                     NodeGene(node_id=5, node_type='hidden'),
                     NodeGene(node_id=3, node_type='output', bias=0)]

        connection_list = [
            ConnectionGene(input_node=1, output_node=5, innovation_number=1, weight=np.random.randn(), enabled=True),
            ConnectionGene(input_node=2, output_node=4, innovation_number=4, weight=np.random.randn(), enabled=True),
            ConnectionGene(input_node=5, output_node=4, innovation_number=3, weight=np.random.randn(), enabled=True),
            ConnectionGene(input_node=4, output_node=3, innovation_number=2, weight=np.random.randn(), enabled=True)]

        genome = Genome(connections=connection_list, nodes=node_list, key=1)

    def test_disable_connection_if_not_part_of_path(self):
        node_list = [NodeGene(node_id=0, node_type='source'),
                     NodeGene(node_id=4, node_type='hidden'),
                     NodeGene(node_id=2, node_type='output', bias=0)]

        connection_list = [
            ConnectionGene(input_node=0, output_node=2, innovation_number=1, weight=np.random.randn(), enabled=True),
            ConnectionGene(input_node=4, output_node=2, innovation_number=4, weight=np.random.randn(), enabled=True)]

        genome = Genome(connections=connection_list, nodes=node_list, key=1)

        self.assertTrue(genome.connections[4].enabled is False)
        self.assertTrue(genome.connections[1].enabled is True)


class TestGenomeReproduction(unittest.TestCase):

    def setUp(self):
        pass

    def test_genome_remove_node_mutation(self):
        node_list = [NodeGene(node_id=0, node_type='source'),
                     NodeGene(node_id=1, node_type='source'),
                     NodeGene(node_id=2, node_type='output', bias=0),
                     NodeGene(node_id=3, node_type='hidden', bias=0),
                     NodeGene(node_id=4, node_type='hidden', bias=0)]

        connection_list = [
            ConnectionGene(input_node=1, output_node=2, innovation_number=1, weight=np.random.randn(), enabled=False),
            ConnectionGene(input_node=1, output_node=3, innovation_number=2, weight=np.random.randn(), enabled=True),
            ConnectionGene(input_node=3, output_node=2, innovation_number=3, weight=np.random.randn(), enabled=False),
            ConnectionGene(input_node=1, output_node=4, innovation_number=4, weight=np.random.randn(), enabled=True),
            ConnectionGene(input_node=3, output_node=4, innovation_number=5, weight=np.random.randn(), enabled=True),
            ConnectionGene(input_node=4, output_node=2, innovation_number=6, weight=np.random.randn(), enabled=True)]

        genome = Genome(connections=connection_list, nodes=node_list, key=3)
        genome.remove_node(node_to_remove=4)
        self.assertTrue(genome.check_num_paths(only_add_enabled_connections=True) == 0)

        for i in range(100):
            genome_2 = Genome(connections=connection_list, nodes=node_list, key=3)
            genome_2.remove_node()
            self.assertTrue(genome_2.check_num_paths(only_add_enabled_connections=True) > 0)


if __name__ == '__main__':
    unittest.main()
