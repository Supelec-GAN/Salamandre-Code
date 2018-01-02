import numpy as np
from brain.network import Network
from fonction import Sigmoid, MnistTest, Norm2
from mnist import MNIST
from engine import Engine
from dataInterface import DataInterface
from ganGame import GanGame
from ganPlot import GanPlot
import matplotlib.pyplot as plt
import os


"""
    Initialisation des données de Mnist
"""
data_interface = DataInterface('GanMnist')  # Non du sous-dossier d'enregistrement des fichiers

param = data_interface.read_conf('config.ini', 'GanMnist')  # Lecture du fichier de config,
# dans la session [GanMnist]

mndata = MNIST(param['file'])  # Import des fichier de Mnist (le paramètre indique l'emplacement)

training_images, training_labels = mndata.load_training()  
training_images = np.array(training_images)/256  # Normalisation de l'image (pixel entre 0 et 1)
number_to_draw = param['number_to_draw']

"""
    On ne conserve dans le set que les 'number_to_draw' du config
"""
not_right_nb = []
for i in range(len(training_images)):
    if training_labels[i] != number_to_draw:
        not_right_nb += [i]
training_images = np.delete(training_images, not_right_nb, axis=0) # A proprifier plus tard,
# c'est pas opti le delete


"""
Initialisation du discriminator
"""
disc_learning_ratio = param['disc_learning_ratio']  # Pour chaque partie, nombre d'apprentissage
# du discriminant sur image réelle
disc_fake_learning_ratio = param['disc_fake_learning_ratio']  # Pour chaque partie,
# nombre d'apprentissage du discriminant sur image fausse, !!!  sans apprentissage du génerateur !!!


disc_activation_funs = np.array(param['disc_activation_funs'])
disc_error_fun = param['disc_error_fun']

discriminator = Network(param['disc_network_layers'], disc_activation_funs, disc_error_fun)

eta_disc = param['eta_disc']


training_fun = param['training_fun']()  # Function donnant la réponse à une vrai image attendu (1
#  par défaut)


"""
Initialisation du generator
"""
generator_layers_neuron_count = param['generator_network_layers']
generator_layers_activation_function = np.array(param['generator_activation_funs'])
generator_error_function = param['generator_error_fun']

generator = Network(generator_layers_neuron_count,
                    generator_layers_activation_function,
                    generator_error_function)

eta_gen = param['eta_gen']

gen_learning_ratio = param['gen_learning_ratio']  # Pour chaque partie, nombre d'apprentissage du
#  discriminant sur image réelle


"""
initialisation de la partie
"""

ganGame = GanGame(discriminator,
                  training_images,
                  training_fun,
                  generator,
                  eta_gen,
                  eta_disc,
                  disc_learning_ratio,
                  gen_learning_ratio,
                  disc_fake_learning_ratio)

play_number = param['play_number']  # N Nombre de partie  (Une partie = i fois apprentissage
# discriminateur sur vrai image, j fois apprentissage génerateur+ discriminateur et
# potentiellement k fois discriminateur avec fausse image

gan_plot = GanPlot('SalamandreGan', number_to_draw, play_number)

"""
Préparation de la sauvegarde des scores du discriminateur pour des vrais images et des images de synthèses
"""
discriminator_real_score = []
discriminator_fake_score = []
test_period = param['test_period']

a, b = ganGame.testDiscriminatorLearning(10)  # Valeur pour le réseau vierge
discriminator_real_score.append(a)
discriminator_fake_score.append(b)

for i in range(play_number):
    ganGame.playAndLearn()
    if i % test_period == 0:
        a, b = ganGame.testDiscriminatorLearning(10)  # effectue n tests et renvoie la moyenne
        # des scores
        discriminator_real_score.append(a)
        discriminator_fake_score.append(b)

data_interface.save(discriminator_real_score, 'discriminator_real_score')  # sauvegarde des
# courbes de score
data_interface.save(discriminator_fake_score, 'discriminator_fake_score')


image_test, associate_noise = ganGame.generateImage()  # génération d'une image à la fin de
# l'apprentissage

gan_plot.save(image_test)


if os.name == 'nt':     # pour exécuter l'affichage uniquement sur nos ordis, et pas la vm
    gan_plot.plot(image_test) # afichage des courbes, commentez à partir de là pour lancement
    # sur VM
    plt.plot(discriminator_real_score)
    plt.plot(discriminator_fake_score)
    plt.show()

