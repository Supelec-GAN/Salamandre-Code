import numpy as np
from brain.network import Network
from fonction import Sigmoid, MnistTest, Norm2
from mnist import MNIST
from engine import Engine
from dataInterface import DataInterface
from ganGame import GanGame
import matplotlib.pyplot as plt


"""
Initialisation du Discriminateur, similaire au Mnist
"""

data_interface = DataInterface('GanMnist')

param = data_interface.read_conf('config.ini', 'GanMnist')

mndata = MNIST(param['file'])
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_testing()

training_images = np.array(training_images)
training_labels = np.array(training_labels)
training_size = param['training_size']

testing_images = np.array(testing_images)
testing_labels = np.array(testing_labels)
testing_size = param['testing_size']

not_sixes = []
for i in range(len(training_images)):
    if training_labels[i] != 6:
        not_sixes += [i]
training_images = np.delete(training_images, not_sixes, axis=0)
training_labels = np.delete(training_labels, not_sixes)

not_sixes = []
for i in range(len(testing_images)):
    if testing_labels[i] != 6:
        not_sixes += [i]
testing_images = np.delete(testing_images, not_sixes, axis=0)
testing_labels = np.delete(testing_labels, not_sixes)

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

play_number = param['play_number']



discriminator_score = np.zeros(play_number)

for i in range(play_number):
    discriminator_score[i] = ganGame.playAndLearn()

image_test, associate_noise = ganGame.generateImage()

image = np.reshape(image_test, [28, 28])
print(image)
plt.imshow(image, cmap='Greys',  interpolation='nearest')
plt.savefig('blkwht.png')

