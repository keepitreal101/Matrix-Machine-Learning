# Libraries
import numpy as np
import random


# Some Functions Necessary
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x)*(1 - sigmoid(x))


# Constants Used
adjustment_rate = 0.95


class Matrix(object):

    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(x, 1) for x in layers[1:]]

    # returns the result of the network
    def forward(self, inp):
        current = inp
        for w, b in zip(self.weights, self.biases):
            current = sigmoid(np.dot(w, current) + b)
        return current

    # returns the correct number of images classified
    def evaluate(self, test_data):
        results = [(np.argmax(self.forward(x)), y) for x, y in test_data]
        total = 0
        for x, y in results:
            if x == y:
                total += 1
        return total

    # collapses the rows of a 2d matrix into one column vector
    def shrink(self, matrix):
        vector = np.random.randn(len(matrix), 1)
        for i in range(len(matrix)):
            tot = 0.0
            for x in matrix[i]:
                tot += x
            vector[i] = tot
        return vector

    # main function
    def learn(self, training, epochs, mini_batch_size, rate, test_data):
        # Assumes that test_data is provided
        test_data = list(test_data)
        test_len = len(test_data)

        n_len = len(training)
        for k in range(epochs):
            # Store Previous Values
            prev_weights = self.weights
            prev_biases = self.biases
            prev_correct = self.evaluate(test_data)

            random.shuffle(training)
            all_batches = [training[k:k + mini_batch_size] for k in range(0, n_len, mini_batch_size)]
            for mini in all_batches:
                self.update_mini(mini, rate)
            current_correct = self.evaluate(test_data)
            if not test_data:
                print("Epoch {}: Complete, No Test Data Provided".format(k))
            else:
                if current_correct < prev_correct:
                    self.weights = [(w + pw) / 2 for w, pw in zip(self.weights, prev_weights)]
                    self.biases = [(b + pb) / 2 for b, pb in zip(self.biases, prev_biases)]
                print("Epoch {}: Precision {} / {}".format(k, self.evaluate(test_data), test_len))

    # backpropogation algorithm
    def matrix_back(self, goes, expect):
        matrix_weight = [np.zeros(b.shape) for b in self.biases]
        vector_biases = [np.zeros(w.shape) for w in self.weights]
        # forward
        activation = goes
        activations = [goes]  # list to store all activation matrices
        zs_matrix = []  # list to store all the intermediate matrices
        for w, b in zip(self.weights, self.biases):
            b_matrix = b
            z = np.dot(w, activation) + b_matrix
            activation = sigmoid(z)
            zs_matrix.append(z)
            activations.append(activation)

        # backward pass
        error_last = (activations[-1] - expect) * \
            sigmoid_prime(zs_matrix[-1])
        collapse = self.shrink(error_last)
        vector_biases[-1] = collapse
        matrix_weight[-1] = np.dot(error_last, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs_matrix[-l]
            sp = sigmoid_prime(z)
            error_last = np.dot(self.weights[-l+1].transpose(), error_last) * sp
            vector_biases[-l] = self.shrink(error_last)
            matrix_weight[-l] = np.dot(error_last, activations[-l-1].transpose())
        return matrix_weight, vector_biases

    # updates over each mini-batch
    def update_mini(self, batch, rate):
        batch_len = len(batch)
        change_weight = [np.zeros(w.shape) for w in self.weights]
        change_biases = [np.zeros(b.shape) for b in self.biases]

        # Changes Batch into a Matrix
        matrix_in, matrix_expected = batch[0]
        for x, y in batch[1:]:
            matrix_in = np.column_stack((matrix_in, x))
            matrix_expected = np.column_stack((matrix_expected, y))

        # Adjust Weights and Biases
        matrix_weight, matrix_biases = self.matrix_back(matrix_in, matrix_expected)
        change_weight = [cw + mw for cw, mw in zip(change_weight, matrix_weight)]
        change_biases = [cb + mb for cb, mb in zip(change_biases, matrix_biases)]
        self.weights = [w - ((rate/batch_len) * cw) for w, cw in zip(self.weights, change_weight)]
        self.biases = [b - ((rate/batch_len) * cb) for b, cb in zip(self.biases, change_biases)]
