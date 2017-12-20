import numpy as np
from brain.network import Network
from fonction import Sigmoid, MnistTest, Norm2
from mnist import MNIST
from engine import Engine
from dataInterface import DataInterface
from ganGame import GanGame


"""
Initialisation du Discriminateur, similaire au Mnist
"""

data_interface = DataInterface('GanMnist')

param = data_interface.read_conf('GanMnist')

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

net = Network(param['network_layers'], activation_funs, error_fun)

eta = param['eta']


training_fun = param['training_fun'](training_labels)
testing_fun = param['testing_fun'](testing_labels)


##
# Il Faudrait supprimer ça et mettre plutot une fonction abstraite
##
def success_fun(o, eo):
    omax = np.max(o)
    if omax == np.dot(np.transpose(o), eo):
        return 1
    return 0

discriminator = Engine(net,
    eta,
    training_images[0:training_size] / 256,
    training_fun,
    testing_images[0:testing_size] / 256,
    testing_fun,
    success_fun,
    learning_iterations,
    test_period,
    randomize_learning_set
    )


"""
Initialisation du discriminator
"""
generator_layers_neuron_count  = param['generator_network_layers']
generator_layers_activation_function = np.array(param['generator_activation_funs'])
generator_error_function = param['generator_error_fun']

generator = Network(generator_layers_neuron_count, generator_layers_activation_function, generator_error_function)


learning_ratio = param['learning_ratio']

ganGame = GanGame(discriminator, generator, learning_ratio)

discriminator_score = np.zeros(play_number)

for i in range(play_number):
    discriminator_score[i] = ganGame.playAndLearn()

final test = ganGame.generateImage()
