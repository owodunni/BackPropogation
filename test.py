from backpropogation import *
import numpy as np


# Testing my sigmoids
def test_sigmoid_0():
    zeros = np.array([0, 0, 0])
    half = np.array([0.5, 0.5, 0.5])
    np.testing.assert_array_equal(sigmoid(zeros), half)


def test_sigmoid_inf():
    assert sigmoid(100) <= 1


def test_sigmoid_negative_inf():
    assert sigmoid(-100) >= 0


# Testing my cost
def test_cost_should_be_zero():
    test_vec = np.array([1, 2, 3])
    assert cost_function(test_vec, test_vec) == 0


def test_cost_should_be_zero():
    labels = np.array([1, 1, 1])
    output = np.array([0, 0, 0])
    assert cost_function(labels, output) == 3 / 2

def test_weighted_input():
    weights = np.eye(2, dtype=int)
    input_ = np.array([1, 0])
    bias = np.array([1, 2])
    result = np.array([2, 2])

    np.testing.assert_array_equal(weighted_input_function(weights, input_, bias), result)

def test_neuron_layer():
    weights = np.eye(2, dtype=int)
    input_ = np.array([1, 0])
    bias = np.array([-1, 0])
    result = np.array([0.5, 0.5])

    np.testing.assert_array_equal(neuron(weights, input_, bias), result)


def test_neuron_layer():
    myNN = NeuronLayer(2, 4)
    output = myNN.input(np.array([1, 2]))
    print(output)
