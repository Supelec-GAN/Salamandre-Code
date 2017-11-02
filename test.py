import numpy as np
from brain.network import Network
from fonction import Sigmoid, MnistTest, Norm2
from mnist import MNIST
from engine import Engine
from dataInterface import DataInterface


mndata = MNIST('.\data')
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_testing()

training_images = np.array(training_images)
training_labels = np.array(training_labels)
training_size = 1000

testing_images = np.array(testing_images)
testing_labels = np.array(testing_labels)
testing_size = 10

learning_iterations = 1
test_period = 100
randomize_learning_set = True


activation_funs = np.array([Sigmoid(0.1), Sigmoid(0.1), Sigmoid(0.1)])
error_fun = Norm2()

net = Network([784, 10, 10, 10], activation_funs, error_fun)

eta = 0.2


# def training_fun(n):
#     r = np.zeros(10)
#     r[training_labels[n]] = 1
#     r = np.reshape(r, (10,1))
#     return r


# def testing_fun(n):
#     r = np.zeros(10)
#     r[testing_labels[n]] = 1
#     r = np.reshape(r, (10,1))
#     return r
training_fun = MnistTest(training_labels)
testing_fun = MnistTest(testing_labels)


engine = Engine(net, eta, training_images[0:1000] / 256, training_fun, testing_images[0:10] / 256,
                testing_fun, 3)

error_during_learning = engine.run()

data_interface = DataInterface('Mnist_debug')

data_params = np.array([learning_iterations, test_period, training_size, testing_size, eta])
param_description = '[learning_iterations, test_period, training_size, testing_size, eta] + Network([784, 10, 10, 10], [Sigmoid(0.1), Sigmoid(0.1), Sigmoid(0.1)]'
data_interface.save(error_during_learning, 'error_during_learning', data_params, param_description)
