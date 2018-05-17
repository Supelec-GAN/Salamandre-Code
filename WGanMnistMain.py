import numpy as np
from brain.network import Network
import dataLoader
from dataInterface import DataInterface
from ganGame import GanGame, WGanGame
from ganPlot import GanPlot
# import matplotlib.pyplot as plt
# import os


# Récupération des paramètres du config.ini
data_interface = DataInterface()

param_liste = data_interface.read_conf('config.ini',
                                       'GanMnist')  # Lecture du fichier de config dans la
# session [GanMnist]
param_desc_disc_liste = data_interface.read_conf('config_algo_descente.ini',
                                                 'Param de desc du disc')
param_desc_gen_liste = data_interface.read_conf('config_algo_descente.ini',
                                                'Param de desc du gen')


# Initialisation des données pour l'apprentissage
training_images, training_labels, _, _ = dataLoader.load_data(param_liste['file'][0],
                                                              param_liste['dataset'][0])

number_exp = param_liste['number_exp'][0]

for exp in range(number_exp):

    print("Lancement de l'experience n°", exp)

    param = data_interface.extract_param(param_liste, exp)

    param_desc_disc = data_interface.extract_param(param_desc_disc_liste, exp)
    param_desc_gen = data_interface.extract_param(param_desc_gen_liste, exp)
    numbers_to_draw = param['numbers_to_draw']

# On ne conserve dans le set que les 'numbers_to_draw' du config
    not_right_nb = []
    for i in range(len(training_images)):
        if training_labels[i] not in numbers_to_draw:
            not_right_nb += [i]
    training_images_exp = np.delete(training_images, not_right_nb, axis=0)  # A proprifier plus
    # tard, c'est pas opti le delete

    batch_size = param["batch_size"]

# Initialisation du dossier de sauvegarde
    save_folder = param['save_folder']
    data_interface = DataInterface(save_folder)

# Initialisation de l'interface d'affichage et de sauvegarde des données des résultat du GAN
    gan_plot = GanPlot(save_folder, numbers_to_draw)

# Initialisation du discriminator
    disc_layers_params = param['disc_network_layers']
    disc_error_fun = param['disc_error_fun']
    disc_error_fun.vectorize()
    gen_error_fun = param['gen_error_fun']
    gen_error_fun.vectorize()
    
    critic = Network(layers_parameters=disc_layers_params,
                            error_function=disc_error_fun,
                            error_gen=gen_error_fun,
                            param_desc='Param de desc du disc',
                            learning_batch_size=batch_size,
                            nb_exp=exp
                            )

    critic_learning_ratio = param['disc_learning_ratio']  # Pour chaque partie, nombre
    # d'apprentissage du discriminant sur image réelle

# Initialisation du generator
    gen_layers_params = param['generator_network_layers']

    generator = Network(layers_parameters=gen_layers_params,
                        error_function=disc_error_fun,
                        param_desc='Param de desc du gen',
                        learning_batch_size=batch_size,
                        nb_exp=exp
                        )

    gen_learning_ratio = param['gen_learning_ratio']  # Pour chaque partie, nombre d'apprentissage
    # du discriminant sur image réelle

# Initialisation de la partie
    training_fun = param['training_fun']  # Function donnant la réponse à une vrai image attendu
    # (1 par défaut)

    ganGame = WGanGame(critic=critic,
                       learning_set=training_images_exp,
                       learning_fun=training_fun,
                       generator=generator,
                       critic_learning_ratio=critic_learning_ratio,
                       gen_learning_ratio=gen_learning_ratio,
                       batch_size=batch_size)

    play_number = param['play_number']  # Nombre de partie  (Une partie = i fois apprentissage
    # discriminateur sur vrai image, j fois apprentissage génerateur+ discriminateur et
    # potentiellement k fois discriminateur avec fausse image

# Préparation de la sauvegarde des scores du discriminateur pour des vrais images et des images de
# synthèses
    score = []
    score_std = []

# Initialisation des paramètres
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

    # Calculation of Wassertstein distance before the game start
    for i in range(200):
        ganGame.critic_learning()
        if (i%10)==0:
            a, b = ganGame.test_critic_learning(lissage_test)  # effectue n test et
            # renvoie la moyenne des scores
    print("on passe au gen")
    for i in range(100):
        ganGame.generator_learning()
        a, b = ganGame.test_critic_learning(lissage_test)  # effectue n test et
        # renvoie la moyenne des scores
        print("Score : ", a)
        print("b", b)

    try:
        for i in range(play_number):
            if i % 10 == 0:
                print("Progress : ", i)
            if i % test_period == 0:
                print("i", i)
                a, b = ganGame.test_critic_learning(lissage_test)  # effectue n test et
                # renvoie la moyenne des scores
                score.append(a)
                print("Score : ", a)
                score_std.append(b)
            if i % image_evolution_number == 0:
                a, b = ganGame.test_critic_learning(lissage_test)
                images_evolution = ganGame.generate_image_test()
                gan_plot.save_multiple_output(images_evolution, str(numbers_to_draw) +
                                              "_au_rang_" + str(i), str(i), a, b)
            learn = ganGame.play_and_learn()
    except KeyboardInterrupt:
        pass

    images_finales = ganGame.generate_image_test()  # génération d'images à la fin
    gan_plot.save_multiple_output(images_finales, str(numbers_to_draw) + str(play_number),
                                      str(play_number), score[-1],
                                      score_std[-1])

    conf = data_interface.save_conf('config.ini', 'GanMnist')  # récupération de la
    # configuration pour la sauvegarde dans les fichiers
    data_interface.save(score, 'score', conf)  # Sauvegarde
    # des courbes de score
    data_interface.save(score_std, 'score_std', conf)

    gan_plot.save_courbes(param, param_desc_gen, param_desc_disc,
                          score, score_std)

    state = generator.save_state()
    gan_plot.save_plot_network_state(state)