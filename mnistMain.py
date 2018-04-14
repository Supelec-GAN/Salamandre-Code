import numpy as np
from brain.network import Network
from fonction import Sigmoid, MnistTest, Norm2
from mnist import MNIST
from engine import Engine
from dataInterface import DataInterface
from errorGraphs import ErrorGraphs


data_interface = DataInterface('Mnist_debug')

param = data_interface.read_conf()


# Chargement des données de MNIST
mndata = MNIST(param['file'])
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_testing()

# Configuration des images d'entrainement
training_images = np.array(training_images)
training_labels = np.array(training_labels)
training_size = param['training_size']

# Configuration des images de test
testing_images = np.array(testing_images)
testing_labels = np.array(testing_labels)
testing_size = param['testing_size']


# Chargement des paramètres de gestion de l'apprentissage
learning_iterations = param['learning_iterations']
test_period = param['test_period']
randomize_learning_set = param['learning_iterations']

# Chargement des fonctions utilisées
activation_funs = np.array(param['activation_funs'])
error_fun = param['error_fun']

# Chargement des paramètres d'apprentissage
eta = param['eta']
learning_set_pass_nb = param['learning_set_pass_nb']
batch_size = param['batch_size']


couche0 = {'type': 'C',
           'activation_function': Sigmoid(0.1),
           'input_size': (28, 28),
           'output_size': (24, 24),
           'filter_size': (5, 5),
           'input_feature_maps': 1,
           'output_feature_maps': 6,
           'convolution_mode': 'valid'}

couche1 = {'type': 'C',
           'activation_function': Sigmoid(0.1),
           'input_size': (24, 24),
           'output_size': (20, 20),
           'filter_size': (5, 5),
           'input_feature_maps': 6,
           'output_feature_maps': 16,
           'convolution_mode': 'valid'}

couche2 = {'type': 'N',
           'activation_function': Sigmoid(0.1),
           'input_size': 784,
           'output_size': 300,
           'noise_size': 0}

couche3 = {'type': 'N',
           'activation_function': Sigmoid(0.1),
           'input_size': 300,
           'output_size': 10,
           'noise_size': 0}

layers_params = [couche2, couche3]

net = Network(layers_params,
              error_function=error_fun,
              learning_batch_size=batch_size)

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
