import numpy as np
import math
import matplotlib.pyplot as plt


class LazyProperty:

    def __init__(self, func):
        self._func = func
        self.__name__ = func.__name__

    def __get__(self, instance, owner):
        if instance is None:
            return None
        result = instance.__dict__[self.__name__] = self._func(instance)

        return result


class Model:

    def __init__(self, train_data, train_labels, test_data, test_labels, batch_size, learning_rate, keep_prob,
                 num_layers,
                 num_epochs, layer_sizes):

        # Data
        self.x_train = train_data  # REMEMBER YOU WANT THIS TO CONTAIN TEST DATA AS WELL
        self.y_train = train_labels
        self.x_test = test_data
        self.y_test = test_labels

        # Network architecture
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.layer_sizes = layer_sizes

        # Parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.parameters = {}  # Empty dictionary to hold weights and biases once initialised
        self.initialise_params()  # initialise parameters

        # Hold data
        self.epoch_cost = []
        self.epoch_counter = []

    @staticmethod
    def relu(x):
        return x * (x > 0)

    @staticmethod
    def sigmoid(x):

        activation = 1 / (1 + np.exp(-x))

        return activation

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""

        sum_ = np.sum(np.exp(x), axis=1)
        sum_.shape = (sum_.shape[0], 1)

        return np.exp(x) / sum_

    @staticmethod
    def xavier_initalizer(num_inputs, num_outputs):
        """
        NOTE: if using RELU then use constant 2 instead of 1 for sqrt
        """
        np.random.seed(7)
        weights = np.random.randn(num_inputs, num_outputs) * np.sqrt(1 / num_inputs)

        return weights

    @staticmethod
    def sigmoid_gradient(a):

        gradient = (a * (1 - a))

        return gradient

    @staticmethod
    def relu_gradient(x):
        return 1. * (x > 0)

    @staticmethod
    def cost_fn(logits, labels, n_examples, loss_type='single_class'):

        if loss_type is 'single_class':
            # Define cost function
            loss = -((labels * np.log(logits)) + ((1 - labels) * np.log(1 - logits)))
            cost_ = (1 / n_examples) * np.sum(loss + eps, axis=0)

        elif loss_type is 'multi_class':
            loss = -(np.sum((labels*np.log(logits))))
            cost_ = np.sum(loss) * (1/n_examples)

        return cost_

    def plot_graph(self):
        plt.plot(self.epoch_counter, self.epoch_cost)
        plt.show()

    @staticmethod
    def batchnorm_forward(x, gamma, beta):
        # compute per-features mean and std_deviation
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        # normalize and zero-center (explicit for caching purposes)
        x_mu = x - mean
        inv_var = 1.0 / np.sqrt(var + eps)
        x_hat = x_mu * inv_var

        # squash
        out = gamma * x_hat + beta

        # cache variables for backward pass
        cache = x_mu, inv_var, x_hat, gamma

        return out, cache

    @staticmethod
    def batchnorm_backward(dout, cache):

        N, D = dout.shape
        x_mu, inv_var, x_hat, gamma = cache

        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(x_hat * dout, axis=0)
        dx_hat = np.dot(dout, gamma.T)

        dvar = np.sum((dx_hat * x_mu * (-0.5) * inv_var ** 3), axis=0)
        dmu = (np.sum((dx_hat * -inv_var), axis=0)) + (dvar * (-2.0 / N) * np.sum(x_mu, axis=0))

        dx1 = dx_hat * inv_var
        dx2 = dvar * (2.0 / N) * x_mu
        dx3 = (1.0 / N) * dmu

        # final partial derivatives
        dx = dx1 + dx2 + dx3

        return dx, dgamma, dbeta

    def bias_correction(self, variable_name, timestep, moment=None):
        assert (isinstance(variable_name, str))
        if moment == 1:
            bias_corrected = self.parameters['moment1_{}'.format(variable_name)] / (1 - beta1 ** timestep)

        elif moment == 2:
            bias_corrected = self.parameters['moment2_{}'.format(variable_name)] / (1 - beta2 ** timestep)
        else:
            raise ValueError

        return bias_corrected

    def update_moment(self, variable_name, gradient, moment=None):
        assert (isinstance(variable_name, str))
        if moment == 1:
            self.parameters['moment1_{}'.format(variable_name)] = \
                (beta1 * self.parameters['moment1_{}'.format(variable_name)]) + ((1 - beta1) * gradient)

        if moment == 2:
            self.parameters['moment2_{}'.format(variable_name)] = \
                (beta2 * self.parameters['moment2_{}'.format(variable_name)]) + ((1 - beta2) * np.power(gradient, 2))

    def create_layer(self, input_matrix, weights_matrix, bias_matrix, activation_fn='relu', batch_norm=False):

        assert(isinstance(activation_fn, str))

        z = np.dot(input_matrix, weights_matrix) + bias_matrix
        # print(z, 'zzzzzzzzzzzzzzzzzz')
        if batch_norm is True:
            gamma = self.parameters['gamma1']
            beta = self.parameters['beta1']
            z, cache = self.batchnorm_forward(z, gamma, beta)

        if activation_fn is 'sigmoid':
            activation = self.sigmoid(z)
        elif activation_fn is 'relu':
            activation = self.relu(z)
        elif activation_fn is 'softmax':
            activation = self.softmax(z)
        # print(activation, 'aaaaaaaaaaa')
        if batch_norm is True:
            return activation, cache
        else:
            return activation

    def initialise_params(self):

        if self.layer_sizes[0, 0] != self.x_train.shape[1]:
            raise ValueError('Number of inputs must match first entry in layer_sizes')

        if self.layer_sizes.shape[0] != self.num_layers:
            raise ValueError('Number of layers defined must be equal to number of layers set')

        if self.layer_sizes.shape[1] != 1:
            raise ValueError('layer_sizes must be a row vector')

        # Iterate through each layer to find num inputs and outputs
        for i in range(self.num_layers - 1):
            index = i + 1  # Keep indexing of parameters to begin from 1 for convenience
            num_inputs = self.layer_sizes[i, 0]  # Current layer number of inputs
            num_outputs = self.layer_sizes[i + 1, 0]  # Next layer expected number of inputs

            self.parameters['w{}'.format(index)] = self.xavier_initalizer(num_inputs, num_outputs)

            # Initialise moment vectors
            self.parameters['moment1_w{}'.format(index)] = 0
            self.parameters['moment2_w{}'.format(index)] = 0

            self.parameters['moment1_b{}'.format(index)] = 0
            self.parameters['moment2_b{}'.format(index)] = 0

            # Only need one bias for last layer
            if i == self.num_layers - 2:
                self.parameters['b{}'.format(index)] = np.zeros((1, n_classes))

            else:
                self.parameters['b{}'.format(index)] = np.zeros((1, num_outputs))

            if i != 0:
                # Parameters for batch norm
                n_parameters = self.layer_sizes[1, 0]  # Number of neurons on second layer
                self.parameters['gamma{}'.format(index - 1)] = np.ones((1, n_parameters))
                self.parameters['moment1_gamma{}'.format(index - 1)] = 0
                self.parameters['moment2_gamma{}'.format(index - 1)] = 0

                self.parameters['beta{}'.format(index - 1)] = np.zeros((1, n_parameters))
                self.parameters['moment1_beta{}'.format(index - 1)] = 0
                self.parameters['moment2_beta{}'.format(index - 1)] = 0

        return self.parameters

    def calc_train_accuracy(self, train=True):

        if train is True:
            data = self.x_train
            labels = self.y_train
        else:
            data = self.x_test
            labels = self.y_test

        prediction = self.predict(data, activation_fn_list)

        accuracy = (prediction == labels)  # Returns bool array
        accuracy = accuracy * 1  # Turns bools into ints
        accuracy = np.average(accuracy)

        return accuracy * 100

    def calc_gradients(self, data_batch, labels_batch, n_examples, batch_norm):

        if batch_norm is True:
            prediction, a1, cache = self.predict(data_batch, activation_fn_list, optimise=True)

            # Compute gradients last layer
            dZ2 = (prediction - labels_batch) * (1 / n_examples)
            dW2 = np.dot(a1.T, dZ2)
            dB2 = np.sum(dZ2, axis=0)

            # Compute gradients for batch norm
            if activation_fn_list[0] is 'sigmoid':
                dY = np.dot(dZ2, self.parameters['w2'].T) * self.sigmoid_gradient(a1)
            elif activation_fn_list[0] is 'relu':
                dY = np.dot(dZ2, self.parameters['w2'].T) * self.relu_gradient(a1)

            dZ1, dgamma, dbeta = self.batchnorm_backward(dY, cache)

            # Compute gradients second layer
            dW1 = np.dot(data_batch.T, dZ1)
            dB1 = np.sum(dZ1, axis=0)

            gradients_dict = {'w1': dW1, 'w2': dW2, 'b1': dB1, 'b2': dB2, 'beta1': dbeta, 'gamma1': dgamma}

        elif batch_norm is False:
            prediction, a1 = self.predict(data_batch, activation_fn_list, optimise=True)

            # Compute gradients first layer
            dZ2 = (prediction - labels_batch) * (1 / n_examples)
            dW2 = np.dot(a1.T, dZ2)
            dB2 = np.sum(dZ2, axis=0)

            # Compute gradients second layer
            if activation_fn_list[0] is 'sigmoid':
                dZ1 = np.dot(dZ2, self.parameters['w2'].T) * self.sigmoid_gradient(a1)
            elif activation_fn_list[0] is 'relu':
                dZ1 = np.dot(dZ2, self.parameters['w2'].T) * self.relu_gradient(a1)

            dW1 = np.dot(data_batch.T, dZ1)
            dB1 = np.sum(dZ1, axis=0)

            gradients_dict = {'w1': dW1, 'w2': dW2, 'b1': dB1, 'b2': dB2}

        dB2.shape = (1, dB2.shape[0])
        dB1.shape = (1, dB1.shape[0])

        assert(self.parameters['w1'].shape == dW1.shape)
        assert(self.parameters['w2'].shape == dW2.shape)
        assert(self.parameters['b1'].shape == dB1.shape)
        assert(self.parameters['b2'].shape == dB2.shape)

        return gradients_dict, prediction

    def predict(self, current_batch, activation_fn_list, optimise=False):

        # Define parameters
        w1 = self.parameters['w1']
        b1 = self.parameters['b1']
        w2 = self.parameters['w2']
        b2 = self.parameters['b2']

        # Define network and prediction
        if batch_norm is True:

            a1, cache = self.create_layer(current_batch, w1, b1, activation_fn=activation_fn_list[0], batch_norm=True)  # First layer
            prediction = self.create_layer(a1, w2, b2, activation_fn=activation_fn_list[1], batch_norm=False)  # Second layer

        elif batch_norm is False:

            a1 = self.create_layer(current_batch, w1, b1, activation_fn=activation_fn_list[0])  # First layer
            prediction = self.create_layer(a1, w2, b2, activation_fn=activation_fn_list[1])  # Second layer

        if optimise is True and batch_norm is True:
            return prediction, a1, cache  # Gives raw values to calculate loss during training
        if optimise is True and batch_norm is False:
            return prediction, a1
        if optimise is False:
            return np.around(prediction)  # Gives values rounded to 0 or 1 to see prediction result on test set

    def optimise(self):

        timestep = 0

        for batch_start in range(0, self.x_train.shape[0], self.batch_size):

            timestep += 1

            current_batch = self.x_train[batch_start:batch_start + self.batch_size, :]
            current_labels = self.y_train[batch_start:batch_start + self.batch_size, :]

            n_examples = current_batch.shape[0]

            gradients_dict, prediction = self.calc_gradients(current_batch, current_labels, n_examples, batch_norm=batch_norm)

            cost = self.cost_fn(prediction, current_labels, n_examples, loss_type='multi_class')

            # Update parameters
            if adam_optimizer is False:

                for variable in gradients_dict:
                    self.parameters[variable] = self.parameters[variable] - \
                                                (self.learning_rate * gradients_dict[variable])

            elif adam_optimizer is True:

                for variable in gradients_dict:

                    self.update_moment(variable, gradients_dict[variable], 1)
                    self.update_moment(variable, gradients_dict[variable], 2)

                    # Bias correction
                    bias_corr_m1 = self.bias_correction(variable, timestep, 1)
                    bias_corr_m2 = self.bias_correction(variable, timestep, 2)

                    self.parameters[variable] = self.parameters[variable] - \
                                                (self.learning_rate * (bias_corr_m1 / np.sqrt(bias_corr_m2)))

        return cost.item()


# Test and Train data
n_generated = 5000  # How many training examples to be generated
use_onehot = True

if use_onehot is True:
    data_train = np.random.randint(2, size=(n_generated, 2))
    labels_train = np.empty((n_generated, 2))
    for row in range(data_train.shape[0]):
        if data_train[row, 0] == 1 and data_train[row, 1] == 1:
            labels_train[row, 0] = 1
            labels_train[row, 1] = 0
        elif data_train[row, 0] == 0 and data_train[row, 1] == 0:
            labels_train[row, 0] = 1
        else:
            labels_train[row, 0] = 0
            labels_train[row, 1] = 0
            labels_train[row, 1] = 1

    data_test = np.array(
        [[1, 1], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 1], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1],
         [1, 1], [0, 0]])
    labels_test = np.array([[1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0]])

elif use_onehot is False:
    data_train = np.random.randint(2, size=(n_generated, 2))
    labels_train = np.empty((n_generated, 1))
    for column in range(data_train.shape[0]):
        if data_train[column, 0] == 1 and data_train[column, 1] == 1:
            labels_train[column] = 1
        elif data_train[column, 0] == 0 and data_train[column, 1] == 0:
            labels_train[column] = 1
        else:
            labels_train[column] = 0

    data_test = np.array(
        [[1, 1], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 1], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1],
         [1, 1], [0, 0]])
    labels_test = np.array([[1], [0], [1], [1], [0], [0], [1], [1], [1], [0], [1], [1], [0], [0], [1], [1]])
# Parameters
batch_size = 32
batch_norm = False
adam_optimizer = False
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
learning_rate = 0.1
keep_prob = 0.5
num_layers = 3
num_epochs = 125
n_features = data_train.shape[1]
n_classes = 2
activation_fn_list = ['sigmoid', 'sigmoid']  # specify what activation function you for a given layer

# Architecture for network
layer_sizes = np.array([[n_features], [2], [n_classes]])

# Initialise model
model = Model(data_train, labels_train, data_test, labels_test, batch_size, learning_rate, keep_prob, num_layers,
              num_epochs, layer_sizes)

for i in range(model.num_epochs):
    cost = model.optimise()

    # Keep track of costs
    model.epoch_counter.append(i)
    model.epoch_cost.append(cost)

    # Check cost and accuracy at every quarter and last epoch
    if i % (round(num_epochs * 0.25, 0)) == 0 or i == model.num_epochs - 1:
        accuracy = round(model.calc_train_accuracy(train=True), 0)

        print('EPOCH:', i, '\t', 'Cost:', round(cost, 3), '\t',
              'Accuracy:', '%{}'.format(accuracy))

model.plot_graph()

# Test model on validation data
test_prediction = model.predict(model.x_test, activation_fn_list)
accuracy = round(model.calc_train_accuracy(train=False), 0)

print(test_prediction, '\n', 'Test Accuracy:', '%{}'.format(accuracy))

def main():
    pass


if __name__ == "__main__":
    main()