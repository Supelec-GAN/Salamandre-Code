import matplotlib.pyplot as plt
import numpy as np
import os
from time import gmtime, strftime
from matplotlib.gridspec import GridSpec


class ErrorGraphs:

    def __init__(self, name, learning_iterations, eta, net, test_period):
        self._name = name
        self.learning_iterations = learning_iterations
        self.eta = eta
        self.net = net
        self.test_period = test_period

    def save(self, errors_during_learning, param):
        # create directory if it doesn't exist
        if not os.path.exists(self._name):
            os.mkdir(self._name)

        plt.close()
        fig = plt.figure()

        gs = GridSpec(1, 20)
        images = slice(0, param['learning_set_pass_nb'] // param['test_period'],
                       (param['learning_set_pass_nb'] // param['testing_size']) //
                       param['test_period'])

        ax_D_x = fig.add_subplot(gs[0, 0:10])
        ax_D_x.autoscale(axis='x')
        if np.shape(errors_during_learning) == (len(errors_during_learning),):
            data = errors_during_learning
            errorbar = np.zeros(len(errors_during_learning))
        else:
            data = np.mean(errors_during_learning, 0)
            std = np.std(errors_during_learning, 0)

            errorbar = std / np.sqrt(len(errors_during_learning))

        print(np    .shape(data))
        #ax_D_x.plot(np.linspace(0, np.shape(data)[0] -1, np.shape(data)[0]), data, '.-', label='Sortie', markevery=images)



        ax_D_x.set_xlabel("Nombre de parties (X" + str(param['test_period']) + ")")

        ax_D_x.legend(loc=0, bbox_to_anchor=(1.3, 0.1))
        plt.ylabel("Erreur moyenne sur le batch de test pour les " + str(self.learning_iterations)
                   + " runs")
        plt.xlabel("Apprentissages")

        info = fig.add_subplot(gs[0, 10])

        info.set_xticks([])
        info.set_yticks([])
        info.axis('off')

        network = param['network']
        network_layers = str(network[0]["input_size"])
        for i in range(len(param['network'])):
            network_layers += ", " + str(network[i]["output_size"])
        info.text(0.01, 0.83, 'Forme du réseau : ' + network_layers,
                  fontsize=8)
        info.text(0.01, 0.45, 'Eta : ' +
                  str(param['eta']), fontsize=8)

        info.text(0.01, 0.31, "Infos courbe", fontsize=12)
        info.text(0.01, 0.27, 'Nombre de partie : ' + str(param['learning_set_pass_nb']),
                  fontsize=8)
        info.text(0.01, 0.23, 'Test toutes les ' + str(param['test_period']) + ' parties',
                  fontsize=8)
        info.text(0.01, 0.19, 'Moyenne sur ' + str(param['testing_size']) + ' samples par test',
                  fontsize=8)
        info.text(0.01, 0.15, "Echantillons d'images toutes les  " +
                  str(param['learning_set_pass_nb'
                            ''] // param['test_period']) + " parties",
                  fontsize=8)
        plt.title("Evolution de l'erreur, test effectué tous les " + str(self.test_period)
                  + " apprentissages")


        # saves the plot in directory
        save_date = strftime('%Y-%m-%d-%H%M%S', gmtime())
        namefile = str(save_date) + '.png'
        namefile = self._name + "/" + namefile
        plt.savefig(namefile,)
        plt.close()









    def plot(self, errors_during_learning):
        # plots the mean error as a function of time
        if np.shape(errors_during_learning) == (len(errors_during_learning),):
            data = errors_during_learning
            errorbar = np.zeros(len(errors_during_learning))
        else:
            data = np.mean(errors_during_learning, 0)
            std = np.std(errors_during_learning, 0)
            errorbar = std / np.sqrt(len(errors_during_learning))
        plt.figure()
        plt.errorbar(np.arange(0,
                               len(errorbar)*self.test_period,
                               self.test_period
                               ),
                     data,
                     yerr=errorbar,
                     capthick=1,
                     ecolor='k',
                     )
        plt.ylabel("Erreur moyenne sur le batch de test pour les " + str(self.learning_iterations)
                   + " runs")
        plt.xlabel("Apprentissages")
        plt.title("Evolution de l'erreur, test effectué tous les " + str(self.test_period)
                  + " apprentissages")
        plt.suptitle("eta =" + str(self.eta) + "\n" + "Réseau en " + str(self.net))
        # saves the plot in directory
        plt.show()







def save_courbes(self, param, param_desc_gen, param_desc_disc, data_real, data_fake):
    if not os.path.exists(self._name):
        os.mkdir(self._name)
    if not os.path.exists(self._name + '/Images'):
        os.mkdir(self._name + '/Images')

    plt.close()
    fig = plt.figure()

    gs = GridSpec(1, 11)
    images = slice(0, param['play_number'] // param['test_period'],
                   (param['play_number'] // param['nb_images_during_learning']) //
                   param['test_period'])

    ax_D_x = fig.add_subplot(gs[0, 0:10])
    ax_D_x.autoscale(axis='x')
    ax_D_x.plot(data_real, '.-', label='D(x)', markevery=images)

    ax_D_x.plot(data_fake, '.-', label='D(G(z))', markevery=images)
    ax_D_x.set_xlabel("Nombre de parties (X" + str(param['test_period']) + ")")

    ax_D_x.legend(loc=0, bbox_to_anchor=(1.3, 0.1))
    ax_D_x.set_title("Réponse du Discriminateur à des images du set et à des images de "
                     "synthèse")

    info = fig.add_subplot(gs[0, 10])

    info.set_xticks([])
    info.set_yticks([])
    info.axis('off')
    info.text(0.01, 0.95, 'Tentative pour ' + str(param['numbers_to_draw']), fontsize=16)

    info.text(0.01, 0.87, 'Formes des réseau', fontsize=12)
    info.text(0.01, 0.83, 'Forme du générateur : ' + str(param['generator_network_layers']),
              fontsize=8)
    # info.text(0.01, 0.79, 'Bruit du générateur : ' + str(param['noise_layers_size']),
    #           fontsize=8)
    info.text(0.01, 0.75, 'Forme du discriminateur : ' + str(param['disc_network_layers']),
              fontsize=8)

    info.text(0.01, 0.61, 'Ratio D image du set : ' +
              str(param['disc_learning_ratio']), fontsize=8)
    info.text(0.01, 0.57, 'Ratio G et D même image de synthèse : ' +
              str(param['gen_learning_ratio']), fontsize=8)
    info.text(0.01, 0.53, 'Ratio D image de synthèse : ' +
              str(param['disc_fake_learning_ratio']), fontsize=8)
    info.text(0.01, 0.49, 'Ratio G image de synthèse : ' +
              str(param['gen_learning_ratio_alone']), fontsize=8)
    info.text(0.01, 0.45, 'Eta générateur : ' +
              str(param_desc_gen['eta']), fontsize=8)
    info.text(0.01, 0.41, 'Eta discriminateur : ' +
              str(param_desc_disc['eta']), fontsize=8)

    info.text(0.01, 0.31, "Infos courbe", fontsize=12)
    info.text(0.01, 0.27, 'Nombre de partie : ' + str(param['play_number']),
              fontsize=8)
    info.text(0.01, 0.23, 'Test toutes les ' + str(param['test_period']) + ' parties',
              fontsize=8)
    info.text(0.01, 0.19, 'Moyenne sur ' + str(param['lissage_test']) + ' samples par test',
              fontsize=8)
    info.text(0.01, 0.15, "Echantillons d'images toutes les  " +
              str(param['play_number'] // param['nb_images_during_learning']) + " parties",
              fontsize=8)

    save_date = strftime('%Y-%m-%d-%H%M%S', gmtime())
    plt.savefig(self._name + '/Images/' + save_date + 'Courbes' + '.png',
                bbox_inches='tight', dpi=300)  # sauvegarde de l'image
    plt.close()