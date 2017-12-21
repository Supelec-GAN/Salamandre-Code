import numpy as np
from brain.network import Network
from fonction import Sigmoid, MnistTest, Norm2
from mnist import MNIST
from engine import Engine
from dataInterface import DataInterface
from ganGame import GanGame
import matplotlib.pyplot as plt


"""
    Initialisation des données de Mnist
"""
data_interface = DataInterface('GanMnist')  # Non du sous-dossier d'enregistrement des fichiers

param = data_interface.read_conf('config.ini', 'GanMnist')  # Lecture du fichier de config, dans la session [GanMnist]

mndata = MNIST(param['file'])  # Import des fichier de Mnist (le paramètre indique l'emplacement)

training_images, training_labels = mndata.load_training()  
training_images = np.array(training_images)/256  # Normalisation de l'image (pixel entre 0 et 1)

"""
    On ne conserve dans le set que les 6
"""
not_sixes = []
for i in range(len(training_images)):
    if training_labels[i] != 6:
        not_sixes += [i]
training_images = np.delete(training_images, not_sixes, axis=0)


"""
Initialisation du discriminator
"""
disc_learning_ratio = param['disc_learning_ratio']  # Pour chaque partie, nombre d'apprentissage du discriminant sur image réelle
disc_fake_learning_ratio = param['disc_fake_learning_ratio']  # Pour chaque partie, nombre d'apprentissage du discriminant sur image fausse, !!!  sans apprentissage du génerateur !!!


disc_activation_funs = np.array(param['disc_activation_funs'])
disc_error_fun = param['disc_error_fun']

discriminator = Network(param['disc_network_layers'], disc_activation_funs, disc_error_fun)

eta_disc = param['eta_disc']


training_fun = param['training_fun']()  # Function donnant la réponse à une vrai image attendu (1 par défaut)


"""
Initialisation du generator
"""
generator_layers_neuron_count = param['generator_network_layers']
generator_layers_activation_function = np.array(param['generator_activation_funs'])
generator_error_function = param['generator_error_fun']

generator = Network(generator_layers_neuron_count, generator_layers_activation_function, generator_error_function)

eta_gen = param['eta_gen']

gen_learning_ratio = param['gen_learning_ratio']  # Pour chaque partie, nombre d'apprentissage du discriminant sur image réelle


"""
initialisation de la partie
"""

ganGame = GanGame(discriminator, training_images, training_fun, generator, eta_gen, eta_disc, disc_learning_ratio, gen_learning_ratio, disc_fake_learning_ratio)

play_number = param['play_number']  #N Nombre de partie  (Une partie = i fois apprentissage discriminateur sur vrai image, j fois apprentissage génerateur+ discriminateur et potentiellement k fois discriminateur avec fausse image


"""
Préparation de la sauvegarde des scores du discriminateur pour des vrais images et des images de synthèses
"""
discriminator_real_score = []
discriminator_fake_score = []

a, b = ganGame.testDiscriminatorLearning(10)  # Valeur pour le réseau vierge
discriminator_real_score.append(a)
discriminator_fake_score.append(b)

for i in range(play_number):
    ganGame.playAndLearn()
    a, b = ganGame.testDiscriminatorLearning(10)  # effectue n test et renvoie la moyenne des scores
    discriminator_real_score.append(a)
    discriminator_fake_score.append(b)

data_interface.save(discriminator_real_score, 'discriminator_real_score')  #Sauvegarde des courbes de score
data_interface.save(discriminator_fake_score, 'discriminator_fake_score')

plt.plot(discriminator_real_score)  # afichage des courbes
plt.plot(discriminator_fake_score)
plt.show()

image_test, associate_noise = ganGame.generateImage()  # Generation d'une image à la fin de l'apprentissage

image = np.reshape(image_test, [28, 28])

plt.imshow(image, cmap='Greys',  interpolation='nearest')
plt.savefig('blkwht.png')  # sauvgarde de l'image
