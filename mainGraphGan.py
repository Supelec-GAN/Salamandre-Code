import numpy as np
from brain.network import Network
from dataInterface import DataInterface
from errorGraphs import ErrorGraphs
import matplotlib.pyplot as plt


data_interface = DataInterface('GanMNIST')
param_real, data_real = data_interface.load("2018-01-17-234131_discriminator_real_score.csv")
param_fake, data_fake = data_interface.load("2018-01-17-234131_discriminator_fake_score.csv")
# numbers_to_draw = param['numbers_to_draw']


# play_number = param['play_number']


# disc_activation_funs = np.array(param['disc_activation_funs'])

# disc_error_fun = param['disc_error_fun']

# # net = Network(param['network_layers'], activation_funs, error_fun)

# eta = param['eta_disc']
# error_graphs = ErrorGraphs('Mnist_debug_graphes',learning_iterations, eta, param['network_layers'], test_period)

plt.plot(data_real)
plt.plot(data_fake)
plt.show()
# error_graphs.save(data)
