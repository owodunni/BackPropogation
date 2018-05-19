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

    def setWeights(self, W):
        self.weights = W

    def setBias(self, B):
        self.bias = B

    def input(self, input_):
        return neuron(self.weights, input_, self.bias)


if __name__ == '__main__':
    myNN = [NeuronLayer(3, 3),
            NeuronLayer(3, 4),
            NeuronLayer(4, 2)]

    mean1 = (50, 50, 50)
    mean2 = (100, 50, 0)
    cov = [[20, 5, 0],
           [5, 10, 0],
           [0, 0, 10]]
    sample1 = np.random.multivariate_normal(mean1, cov)
    sample2 = np.random.multivariate_normal(mean2, cov)
    print(sample1)
    print(sample2)

    input_ = sample1

    if random.choice([True, False]):
        input_ = sample1
        labels = (1, 0)
    else:
        input_ = sample2
        labels = (0, 1)

    print("* sample")
    print(input_)

    for layer in myNN:
        input_ = layer.input(input_)

    print("* cost")
    print(cost_function(labels, input_))
    print("* result")
    print(input_)
