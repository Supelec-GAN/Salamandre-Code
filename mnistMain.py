import numpy as np
from brain.network import Network
from fonction import Sigmoid, MnistTest, Norm2
from mnist import MNIST
from engine import Engine
from dataInterface import DataInterface
from errorGraphs import ErrorGraphs


data_interface = DataInterface('Mnist_debug')

param = data_interface.read_conf()

mndata = MNIST(param['file'])
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_testing()

training_images = np.array(training_images)
training_labels = np.array(training_labels)
training_size = param['training_size']

testing_images = np.array(testing_images)
testing_labels = np.array(testing_labels)
testing_size = param['testing_size']

learning_iterations = param['learning_iterations']
test_period = param['test_period']
randomize_learning_set = param['learning_iterations']


activation_funs = np.array(param['activation_funs'])
error_fun = param['error_fun']

eta = param['eta']
learning_set_pass_nb = param['learning_set_pass_nb']
batch_size = param['batch_size']

net = Network(param['network_layers'], activation_funs, error_fun, "Parametres de descente", batch_size)

error_graphs = ErrorGraphs('Mnist_debug_graphes', learning_iterations, eta, net, test_period)

momentum = param['momentum']

training_fun = param['training_fun'](training_labels)
testing_fun = param['testing_fun'](testing_labels)


def success_fun(o, eo):
    omax = np.max(o, axis=0)
    a = np.max(o*eo, axis=0)
    res = np.array([omax[i] == a[i] for i in range(len(a))])
    return res
#    if omax == np.dot(np.transpose(o), eo):
#        return 1
#    return 0


engine = Engine(net,
                eta,
                training_images[0:training_size] / 256,
                training_fun,
                testing_images[0:testing_size] / 256,
                testing_fun,
                success_fun,
                momentum,
                learning_iterations,
                test_period,
                learning_set_pass_nb,
                randomize_learning_set)

error_during_learning = engine.run()

data_interface.save(error_during_learning, 'error_during_learning', data_interface.save_conf())

error_graphs.save(error_during_learning)
