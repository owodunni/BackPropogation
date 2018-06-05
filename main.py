from backpropogation import NeuronLayer
from backpropogation import generate_sample


myNN = [NeuronLayer(3, 4),
        NeuronLayer(4, 2)]

i = 0
while i < 100000:

    x, y = generate_sample()
    a = x/255
    for layer in myNN:
        a = layer.input(a)

    myNN[-1].error1(a, y)
    myNN[-2].error2(myNN[-1].weights, myNN[-1].error)

    for layer in myNN:
        layer.update_layer()

    i+=1

i = 0
while i < 10:

    x, y = generate_sample()
    a = x/266
    for layer in myNN:
        a = layer.input(a)

    print("******")
    print("Input")
    print(x/255)
    print("output")
    print(a)
    print("label")
    print(y)
    i+=1

