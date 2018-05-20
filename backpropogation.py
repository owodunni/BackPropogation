import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_delta(x):
    fx = sigmoid(x)
    return np.multiply(fx, (1 - fx))


def cost_function(labels, output):
    return 1 / 2 * (np.sum(np.square(labels - output)))


def weighted_input_function(weights, input_, bias):
    return np.matmul(weights, input_) + bias


def create_weights(l, k):
    return np.matrix(np.random.rand(l, k))


def create_bias(l):
    return np.transpose(np.matrix(np.random.rand(l)))


def bp1(x, y, z):
    a = np.subtract(x, y)
    return np.multiply(a, sigmoid_delta(z))


def bp2(w, error, z):
    return np.multiply(np.matmul(np.transpose(w), error), z)


class NeuronLayer:
    def __init__(self, inputs_, neurons):
        self.weights = create_weights(neurons, inputs_)
        self.bias = create_bias(neurons)
        self.step = 0.1

    def update_layer(self):
        self.delta_w = np.multiply(self.step, np.transpose(np.matmul(self.a_in, np.transpose(self.error))))
        self.weights = self.weights - self.delta_w

        self.delta_b = np.multiply(self.step, self.error)
        self.bias = self.bias - self.delta_b

    def input(self, input_):
        self.z = weighted_input_function(self.weights, input_, self.bias)
        self.a_in = input_
        self.a_out = sigmoid(self.z)
        return self.a_out

    def error1(self, x, y):
        self.error = bp1(x, y, self.z)
        return self.error

    def error2(self, weights, previous_error):
        self.error = bp2(weights, previous_error, self.z)
        return self.error


def generate_sample():
    mean1 = (50, 50, 50)
    mean2 = (100, 50, 0)
    cov = [[20, 5, 0],
           [5, 10, 0],
           [0, 0, 10]]

    x1 = np.transpose(np.matrix(np.random.multivariate_normal(mean1, cov)))
    x2 = np.transpose(np.matrix(np.random.multivariate_normal(mean2, cov)))

    if random.choice([True, False]):
        x = x1
        y = np.transpose(np.matrix([1, 0]))
    else:
        x = x2
        y = np.transpose(np.matrix([0, 1]))
    return x, y


if __name__ == '__main__':
    myNN = [NeuronLayer(3, 3),
            NeuronLayer(3, 4),
            NeuronLayer(4, 2)]

    i = 0
    while i < 100000:
        i += 1

        x, y = generate_sample()
        sample = x
        x = np.multiply(1 / 255.0, x)
        for layer in myNN:
            x = layer.input(x)

        print("\n")
        print("* result")
        print(x)
        print("* label")
        print(y)
        print("* sample")
        print(sample)


        myNN[2].error1(x, y)
        myNN[1].error2(myNN[2].weights, myNN[2].error)
        myNN[0].error2(myNN[1].weights, myNN[1].error)

        for layer in myNN:
            layer.update_layer()
