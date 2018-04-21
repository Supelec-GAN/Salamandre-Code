import numpy as np
from brain.network import Network
import dataLoader
from dataInterface import DataInterface
from ganGame import GanGame
from ganPlot import GanPlot
# import matplotlib.pyplot as plt
# import os

"""
Récupération des paramètres du config.ini
"""

data_interface = DataInterface()
# Lecture du fichier de config dans la session [GanMnist]
param_liste = data_interface.read_conf('config.ini',
                                       'GanMnist')
param_desc_disc_liste = data_interface.read_conf('config_algo_descente.ini',
                                                 'Param de desc du disc')
param_desc_gen_liste = data_interface.read_conf('config_algo_descente.ini',
                                                'Param de desc du gen')

"""
Initialisation des données pour l'apprentissage
"""

training_images, training_labels, _, _ = dataLoader.load_data(param_liste['file'][0],
                                                              param_liste['dataset'][0])

number_exp = param_liste['number_exp'][0]

for exp in range(number_exp):

    print("Lancement de l'experience n°", exp)

    param = data_interface.extract_param(param_liste, exp)

    param_desc_disc = data_interface.extract_param(param_desc_disc_liste, exp)
    param_desc_gen = data_interface.extract_param(param_desc_gen_liste, exp)
    numbers_to_draw = param['numbers_to_draw']

    """
    On ne conserve dans le set que les 'numbers_to_draw' du config
    """
    not_right_nb = []
    for i in range(len(training_images)):
        if training_labels[i] not in numbers_to_draw:
            not_right_nb += [i]
    training_images_exp = np.delete(training_images, not_right_nb, axis=0)  # A proprifier plus
    # tard, c'est pas opti le delete

    batch_size = param["batch_size"]
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

    disc_layers_params = param['disc_network_layers']
    disc_error_fun = param['disc_error_fun']
    disc_error_fun.vectorize()
    gen_error_fun = param['gen_error_fun']
    gen_error_fun.vectorize()

    discriminator = Network(layers_parameters=disc_layers_params,
                            error_function=disc_error_fun,
                            error_gen=gen_error_fun,
                            param_desc='Param de desc du disc',
                            learning_batch_size=batch_size,
                            nb_exp=exp
                            )

    disc_learning_ratio = param['disc_learning_ratio']  # Pour chaque partie, nombre
    # d'apprentissage du discriminant sur image réelle
    disc_fake_learning_ratio = param['disc_fake_learning_ratio']  # Pour chaque partie,
    # nombre d'apprentissage du discriminant sur image fausse, !!!  sans apprentissage du
    # génerateur !!!

    """
    Initialisation du generator
    """

    gen_layers_params = param['generator_network_layers']

    generator = Network(layers_parameters=gen_layers_params,
                        error_function=disc_error_fun,
                        param_desc='Param de desc du gen',
                        learning_batch_size=batch_size,
                        nb_exp=exp
                        )

    gen_learning_ratio = param['gen_learning_ratio']  # Pour chaque partie, nombre d'apprentissage
    # du discriminant sur image réelle
    gen_learning_ratio_alone = param['gen_learning_ratio_alone']

    """
    Initialisation de la partie
    """

    training_fun = param['training_fun']  # Function donnant la réponse à une vrai image attendu
    # (1 par défaut)

    ganGame = GanGame(discriminator=discriminator,
                      learning_set=training_images_exp,
                      learning_fun=training_fun,
                      generator=generator,
                      disc_learning_ratio=disc_learning_ratio,
                      gen_learning_ratio=gen_learning_ratio,
                      disc_fake_learning_ratio=disc_fake_learning_ratio,
                      gen_learning_ratio_alone=gen_learning_ratio_alone,
                      batch_size=batch_size)

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

    try:
        for i in range(play_number):
            learn = ganGame.play_and_learn()
            if i % test_period == 0:
                print("i", i)
                a, b, c, d = ganGame.test_discriminator_learning(lissage_test)  # effectue n test et
                # renvoie la moyenne des scores
                discriminator_real_score.append(a)
                discriminator_fake_score.append(b)
                real_std.append(c)
                fake_std.append(d)
            if i % image_evolution_number == 0:
                a, b, c, d = ganGame.test_discriminator_learning(lissage_test)
                images_evolution = [[]]*nb_images_par_sortie_during_learning
                for j in range(nb_images_par_sortie_during_learning):
                    image, associate_noise = ganGame.generate_image()  # Generation d'une image à
                    # la fin de l'apprentissage
                    images_evolution[j] = image
                if nb_images_par_sortie_during_learning > 0:
                    gan_plot.save_multiple_output(images_evolution, str(numbers_to_draw) +
                                                  "_au_rang_" + str(i), str(i), a, b)
    except KeyboardInterrupt:
        pass

    images_finales = [[]]*final_images
    for i in range(final_images):
        image_test, associate_noise = ganGame.generate_image()  # génération d'une image à la fin de
        # l'apprentissage
        images_finales[i] = image_test
    if final_images > 0:
        gan_plot.save_multiple_output(images_finales, str(numbers_to_draw) + str(play_number),
                                      str(play_number), discriminator_real_score[-1],
                                      discriminator_fake_score[-1])

    conf = data_interface.save_conf('multi_config.ini', 'GanMnist')  # récupération de la
    # configuration pour la sauvegarde dans les fichiers
    data_interface.save(discriminator_real_score, 'discriminator_real_score', conf)  # Sauvegarde
    # des courbes de score
    data_interface.save(discriminator_fake_score, 'discriminator_fake_score', conf)
    data_interface.save(real_std, 'real_std', conf)  # Sauvegarde des courbes de score
    data_interface.save(fake_std, 'fake_std', conf)

    gan_plot.save_courbes(param, param_desc_gen, param_desc_disc,
                          discriminator_real_score, discriminator_fake_score)

    state = generator.save_state()
    gan_plot.save_plot_network_state(state)
    # if os.name == 'nt':     # pour exécuter l'affichage uniquement sur nos ordis, et pas la vm
    #     state = generator.save_state()
    #     gan_plot.plot_network_state(state)

    #     gan_plot.plot_courbes(param, discriminator_real_score, discriminator_fake_score)
