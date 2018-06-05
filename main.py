from backpropogation import NeuronLayer
from backpropogation import generate_sample


myNN = [NeuronLayer(3, 4),
        NeuronLayer(4, 2)]

i = 0
while i < 100:

    x, y = generate_sample()
    a = x
    for layer in myNN:
        a = layer.input(a)

    myNN[-1].error1(a, y)
    myNN[-2].error2(myNN[-1].weights, myNN[-1].error)

    for layer in myNN:
        layer.update_layer()

