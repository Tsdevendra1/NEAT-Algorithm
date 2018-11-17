import unittest
import numpy as np

from genome_neural_network import GenomeNeuralNetwork
from neural_network import ForwardProp, ActivationFunctions, BackProp, NeuralNetwork, create_architecture, create_data
from deconstruct_genome import DeconstructGenome
from genome import Genome
from gene import ConnectionGene, NodeGene


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
            DeconstructGenome.get_node_layers(connections=list(self.genome.connections.values()),
                                              num_nodes=len(self.genome.nodes))[0],
            expected_answer)

    def test_unpack_genome(self):
        expected_answer = np.ones((2, 2))
        answer = DeconstructGenome.unpack_genome(genome=self.genome)['connection_matrices']

        # check that the first layer weights are the correct ones
        self.assertEqual(answer[1].tolist(), expected_answer.tolist())

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

        answer = DeconstructGenome.unpack_genome(genome=genome)['connection_matrices']

        self.assertEqual(answer[1].tolist(), expected_answer.tolist())

    def test_genome_forward_prop(self):
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

        output = ForwardProp.forward_prop(num_layers=genome_nn.num_layers, initial_input=x_data,
                                          layer_weights=genome_nn.weights_dict,
                                          layer_activation_functions=genome_nn.activation_function_dict,
                                          layer_biases=genome_nn.bias_dict, return_number_before_last_activation=True)

        self.assertEqual(expected_answer.tolist(), output.tolist())


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

    def test_add_connection(self):
        expected_number_of_connections = 7
        self.genome.add_connection(new_innovation_number=7)

        self.assertEqual(len(self.genome.connections), expected_number_of_connections)

        new_connection = self.genome.connections[7]
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
        number_of_beginning_connections = len(self.genome.connections)
        expected_number_of_connections = 5
        self.genome.remove_connection()

        self.assertEqual(number_of_beginning_connections, 6)
        self.assertEqual(len(self.genome.connections), expected_number_of_connections)

    def test_add_node(self):
        number_of_beginning_connections = len(self.genome.connections)
        self.assertEqual(number_of_beginning_connections, 6)

        # Because we replaced 1 connection with two
        expected_number_connections = number_of_beginning_connections + 2

        self.genome.add_node(new_innovation_number=7)

        self.assertEqual(len(self.genome.connections), expected_number_connections)

        number_of_disabled_connections = 1

        disabled_counters = 0
        for connection in self.genome.connections.values():
            if not connection.enabled:
                disabled_counters += 1

        # One connection should have been disabled from the node addition
        self.assertEqual(number_of_disabled_connections, disabled_counters)

    def test_remove_node(self):
        pass

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
        child.crossover(genome_1=genome_1, genome_2=genome_2)

        expected_genes = [1, 2, 3, 4, 5, 6]
        actual_genes = []
        for connection in child.connections.values():
            actual_genes.append(connection.innovation_number)

        self.assertEqual(expected_genes, actual_genes)


if __name__ == '__main__':
    unittest.main()
