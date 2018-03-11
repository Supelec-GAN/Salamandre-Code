import numpy as np
from brain.network import Network, GeneratorNetwork, NoisyGeneratorNetwork
from fonction import Sigmoid, MnistTest, Norm2
from mnist import MNIST
from dataInterface import DataInterface
from ganGame import GanGame
import matplotlib.pyplot as plt
from ganPlot import GanPlot
import os

"""
récuperation des paramètres du config.ini
"""

data_interface = DataInterface()  # Non du sous-dossier d'enregistrement des fichiers

param = data_interface.read_conf('config.ini', 'GanMnist')  # Lecture du fichier de config,
# dans la session [GanMnist]


"""
    Initialisation des données de Mnist
"""
mndata = MNIST(param['file'])  # Import des fichier de Mnist (le paramètre indique l'emplacement)

training_images, training_labels = mndata.load_training()  
training_images = np.array(training_images)/256  # Normalisation de l'image (pixel entre 0 et 1)
numbers_to_draw = param['numbers_to_draw']


"""
    On ne conserve dans le set que les 'numbers_to_draw' du config
"""
not_right_nb = []
for i in range(len(training_images)):
    if training_labels[i] not in numbers_to_draw:
        not_right_nb += [i]
training_images = np.delete(training_images, not_right_nb, axis=0)  # A proprifier plus tard,
# c'est pas opti le delete

"""
Initialisation du dossier de sauvegarde
"""
save_folder = param['save_folder']

data_interface = DataInterface(save_folder)
"""
Initialisation de l'interface d'affichage et de sauvegarde des données des résultat du GAN
"""
gan_plot = GanPlot(save_folder, numbers_to_draw)

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
momentum_disc = param['momentum_disc']

training_fun = param['training_fun']()  # Function donnant la réponse à une vrai image attendu (1
# par défaut)


"""
Initialisation du generator
"""
generator_layers_neuron_count = param['generator_network_layers']
noise_layers_size = param['noise_layers_size']
generator_layers_activation_function = np.array(param['generator_activation_funs'])
# generator_error_function = param['generator_error_fun']

generator = NoisyGeneratorNetwork(generator_layers_neuron_count,
                    generator_layers_activation_function,
                    disc_error_fun,
                    noise_layers_size) 

eta_gen = param['eta_gen']
momentum_gen = param['momentum_gen']

gen_learning_ratio = param['gen_learning_ratio']  # Pour chaque partie, nombre d'apprentissage du
#  discriminant sur image réelle
gen_learning_ratio_alone = param['gen_learning_ratio_alone']


"""
initialisation de la partie
"""

ganGame = GanGame(discriminator,
                  training_images,
                  training_fun,
                  generator,
                  eta_gen,
                  eta_disc,
                  momentum_gen,
                  momentum_disc,
                  disc_learning_ratio,
                  gen_learning_ratio,
                  disc_fake_learning_ratio,
                  gen_learning_ratio_alone)

play_number = param['play_number']  # Nombre de partie  (Une partie = i fois apprentissage
# discriminateur sur vrai image, j fois apprentissage génerateur+ discriminateur et
# potentiellement k fois discriminateur avec fausse image


"""
Préparation de la sauvegarde des scores du discriminateur pour des vrais images et des images de
synthèses
"""
discriminator_real_score = []
discriminator_fake_score = []
real_std = []
fake_std = []

"""
Initialisation des paramètres
"""
nb_images_during_learning = param['nb_images_during_learning']
nb_images_par_sortie_during_learning = param['nb_images_par_sortie_during_learning']
test_period = param['test_period']
lissage_test = param['lissage_test']
final_images = param['final_images']
# a, b, c, d = ganGame.testDiscriminatorLearning(10)  # Valeur pour le réseau vierge
# discriminator_real_score.append(a)
# discriminator_fake_score.append(b)
# real_std.append(c)
# fake_std.append(d)

image_evolution_number = play_number//nb_images_during_learning

for i in range(play_number):
    ganGame.playAndLearn()
    if i % test_period == 0:
        print(i)
        a, b, c, d = ganGame.testDiscriminatorLearning(lissage_test)  # effectue n test et renvoie la moyenne
        # des scores
        discriminator_real_score.append(a)
        discriminator_fake_score.append(b)
        real_std.append(c)
        fake_std.append(d)
    if i % image_evolution_number == 0:
        a, b, c, d = ganGame.testDiscriminatorLearning(1)

        for j in range(nb_images_par_sortie_during_learning):
            image, associate_noise = ganGame.generateImage()  # Generation d'une image à la fin de
            # l'apprentissage

            gan_plot.save(image, str(numbers_to_draw) + "_" + str(j) +  "_au_rang_" + str(i),str(i),a, b)


for i in range(final_images):
    image_test, associate_noise = ganGame.generateImage()  # génération d'une image à la fin de
    # l'apprentissage

    gan_plot.save(image_test, str(numbers_to_draw)+ str(i), str(i), discriminator_real_score[-1], discriminator_fake_score[-1])


conf = data_interface.save_conf('config.ini', 'GanMnist')  # récupération de la configuration
# pour la sauvegarde dans les fichiers
data_interface.save(discriminator_real_score, 'discriminator_real_score', conf)  # Sauvegarde des
# courbes de score
data_interface.save(discriminator_fake_score, 'discriminator_fake_score', conf)
data_interface.save(real_std, 'real_std', conf)  # Sauvegarde des courbes de score
data_interface.save(fake_std, 'fake_std', conf)

gan_plot.save_courbes(param, discriminator_real_score, discriminator_fake_score)

if os.name == 'nt':     # pour exécuter l'affichage uniquement sur nos ordis, et pas la vm
    state = generator.save_state()
    gan_plot.plot_network_state(state)

    gan_plot.plot_courbes(param, discriminator_real_score, discriminator_fake_score)
    plt.plot(discriminator_real_score)
    plt.plot(discriminator_fake_score)
