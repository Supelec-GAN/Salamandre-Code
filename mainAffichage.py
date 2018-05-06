import numpy as np
from brain.network import Network
from dataInterface import DataInterface
from errorGraphs import ErrorGraphs


data_interface = DataInterface('Mnist_debug')

param, data = data_interface.load("2017-11-23-162838_error_during_learning.csv")
print(param)
training_size = param['training_size']

testing_size = param['testing_size']

learning_iterations = param['learning_iterations']
test_period = param['test_period']
randomize_learning_set = param['learning_iterations']


activation_funs = np.array(param['activation_funs'])
error_fun = param['error_fun']

# net = Network(param['network_layers'], activation_funs, error_fun)

eta = param['eta']
error_graphs = ErrorGraphs('Mnist_debug_graphes',learning_iterations, eta, param['network_layers'],
                           test_period)

print(type(test_period))
# error_graphs.save(data)

error_graphs.save(data)
