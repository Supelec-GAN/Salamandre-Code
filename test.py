import numpy as np
import matplotlib.pyplot as plt
from xor.network import Network
from fonction import Sigmoid
from mnist import MNIST


mndata = MNIST('.')
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_testing()

training_images = np.array(training_images)
training_labels = np.array(training_labels)


activation_funs = np.array([Sigmoid(1), Sigmoid(1)])

net = Network([784, 1000, 10], activation_funs)

eta = 0.05

training_size = np.size(training_labels)


def reference(n):
    r = [[0] for j in range(10)]
    r[training_labels[n]][0] = 1
    return r


for i in range(10000):
    output = net.compute(training_images[i]/256)
    net.backprop(eta, training_images[i], reference(i))

for i in range(100):
    print(net.compute(testing_images[i]))
    print(testing_labels[i])
