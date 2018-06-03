import random

import mnist_loader
import network
from backpropogation import NeuronLayer
from backpropogation import generate_sample

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

myNN = [NeuronLayer(784, 30),
        NeuronLayer(30, 10)]

net = network.Network([784, 30, 10])

myNN[0].weights = net.weights[0]
myNN[1].weights = net.weights[1]
myNN[0].bias = net.biases[0]
myNN[1].bias = net.biases[1]


random.shuffle(training_data)
test = training_data[0:1]

#net.SGD(training_data, 30, 1, 3.0, test_data=test_data)

#for x, y in test:
#    nabla_b, nabla_w = net.backprop(x,y)

#print("*****layer0")
#print(nabla_b[0][0:2])
#print(nabla_w[0][0:2])
#print("*****layer1")
#print(nabla_b[1][0:2])
#print(nabla_w[1][0:2])

net.update_mini_batch(test, 3.0)
print("weights")
print(net.weights[0][0:2])
print("biases")
print(net.biases[0][0:2])
print("****")
for x,y in test:

    a = x
    for layer in myNN:
        a = layer.input(a)


    myNN[-1].error1(a, y)
    myNN[-2].error2(myNN[-1].weights, myNN[-1].error)

    #print("delta")
    #print(myNN[-1].error[0:1])
    #print(myNN[0].error)

    for layer in myNN:
        layer.update_layer()

    #print("*****layer0")
    #print(myNN[0].nabla_b[0:2])
    #print(myNN[0].nabla_w[0:2])
    #print("*****layer1")
    #print(myNN[1].nabla_b[0:2])
    #print(myNN[1].nabla_w[0:2])

    print("weights")
    print(myNN[0].weights[0:2])
    print("biases")
    print(myNN[0].bias[0:2])

# net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
