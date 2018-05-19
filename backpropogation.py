import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_delta(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def cost_function(labels, output):
    return 1 / 2 * (np.sum(np.square(labels - output)))


def weighted_input_function(weights, input_, bias):
    return np.matmul(weights, input_) + bias


def neuron(weights, input_, bias):
    return sigmoid(weighted_input_function(weights, input_, bias))


def create_weights(l, k):
    return np.random.rand(l, k)


def create_bias(l):
    return np.random.rand(l)


class NeuronLayer:
    def __init__(self, inputs_, neurons):
        self.weights = create_weights(neurons, inputs_)
        self.bias = create_bias(neurons)

    def update_weights(self, w):
        self.weights += w

    def update_bias(self, b):
        self.bias += b

    def input(self, input_):
        return neuron(self.weights, input_, self.bias)


def generate_sample():
    mean1 = (50, 50, 50)
    mean2 = (100, 50, 0)
    cov = [[20, 5, 0],
           [5, 10, 0],
           [0, 0, 10]]

    x1 = np.random.multivariate_normal(mean1, cov)
    x2 = np.random.multivariate_normal(mean2, cov)

    if random.choice([True, False]):
        x = x1
        y = (1, 0)
    else:
        x = x2
        y = (0, 1)

    return x, y


if __name__ == '__main__':
    myNN = [NeuronLayer(3, 3),
            NeuronLayer(3, 4),
            NeuronLayer(4, 2)]



    i = 0
    while i < 1000:
        i += 1

        x, y = generate_sample()
        for layer in myNN:
            x = layer.input(x)

        
