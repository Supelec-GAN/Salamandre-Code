import numpy as np
import matplotlib.pyplot as plt
from xor.network import Network
from fonction import Sigmoid
from mnist import MNIST
from engine import Engine
import csv


mndata = MNIST('.')
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_testing()

training_images = np.array(training_images)
training_labels = np.array(training_labels)

testing_images = np.array(testing_images)
testing_labels = np.array(testing_labels)


activation_funs = np.array([Sigmoid(0.1), Sigmoid(0.1), Sigmoid(0.1)])

net = Network([784, 1000, 300, 10], activation_funs)

eta = 0.2


def training_fun(n):
    r = np.zeros(10)
    r[training_labels[n]] = 1
    r = np.reshape(r, (10,1))
    return r


def testing_fun(n):
    r = np.zeros(10)
    r[testing_labels[n]] = 1
    r = np.reshape(r, (10,1))
    return r


engine = Engine(net, eta, training_images / 256, training_fun, testing_images[0:1000] / 256,
                testing_fun)

engine.run()
