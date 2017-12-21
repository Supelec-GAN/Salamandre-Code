import numpy as np
from brain.network import Network
from dataInterface import DataInterface
from errorGraphs import ErrorGraphs


data_interface = DataInterface('Mnist_debug')

param1, param2, data = data_interface.load_old("2017-11-02-112351_error_during_learning.csv")
print("param1", param1)
print("param2", param2)

training_size = param1[2]

testing_size = param1[3]

learning_iterations = param1[0]
test_period = param1[1]

activation_funs = np.array(param2[1])

# net = Network(param['network_layers'], activation_funs, error_fun)

eta = param1[4]
error_graphs = ErrorGraphs('Mnist_debug_graphes',learning_iterations, eta, param2[0], test_period)

print(type(test_period))
# error_graphs.save(data)

error_graphs.save(data)
