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

    def save(self, errors_during_learning, param, param_algo_descente):
        # create directory if it doesn't exist
        if not os.path.exists(self._name):
            os.mkdir(self._name)

        plt.close()
        fig = plt.figure()

        gs = GridSpec(1, 20)

        ax_D_x = fig.add_subplot(gs[0, 0:10])
        ax_D_x.autoscale(axis='x')
        if np.shape(errors_during_learning) == (len(errors_during_learning),):
            data = errors_during_learning
            errorbar = np.zeros(len(errors_during_learning))
        else:
            data = np.mean(errors_during_learning, 0)
            std = np.std(errors_during_learning, 0)

            errorbar = std / np.sqrt(len(errors_during_learning))

        ax_D_x.plot(np.linspace(0, np.shape(data)[0] -1, np.shape(data)[0]), data, '.-', label = 'Sortie')



        ax_D_x.set_xlabel("Nombre de parties (X" + str(param['test_period']) + ")")

        ax_D_x.legend(loc=0, bbox_to_anchor=(1.3, 0.1))
        plt.ylabel("Succès moyen sur le batch de test pour les " + str(self.learning_iterations)
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


        algo_utilise = param_algo_descente['algo_utilise'][0]
        info.text(0.01, 0.77, 'Algorithme : ' +
                  str(algo_utilise), fontsize=8)
        if algo_utilise == 'Gradient':
            info.text(0.01, 0.73, 'Momentum : ' +
                  str(param_algo_descente['momentum']), fontsize=8)
            info.text(0.01, 0.69, 'Eta : ' +
                  str(param_algo_descente['eta']), fontsize=8)
        elif algo_utilise == 'Adagrad':
            info.text(0.01, 0.73, 'Momentum : ' +
                      str(param_algo_descente['momentum']), fontsize=8)
            info.text(0.01, 0.69, 'Eta : ' +
                     str(param_algo_descente['eta']), fontsize=8)
            info.text(0.01, 0.65, 'Epsilon : ' +
                      str(param_algo_descente['epsilon']), fontsize=8)
        elif algo_utilise == 'RMSProp':
            info.text(0.01, 0.73, 'Momentum : ' +
                      str(param_algo_descente['momentum']), fontsize=8)
            info.text(0.01, 0.69, 'Eta : ' +
                      str(param_algo_descente['eta']), fontsize=8)
            info.text(0.01, 0.65, 'Gamma : ' + str(param_algo_descente['gamma']), fontsize=8)
            info.text(0.01, 0.61, 'Epsilon : ' +
                      str(param_algo_descente['epsilon']), fontsize=8)
            info.text(0.01, 0.57, "Moment d'ordre " + str(param_algo_descente['moment']), fontsize=8)
        elif algo_utilise == 'Adadelta':
            info.text(0.01, 0.73, 'Momentum : ' +
                      str(param_algo_descente['momentum']), fontsize=8)
            info.text(0.01, 0.69, 'Gamma : ' + str(param_algo_descente['gamma']), fontsize=8)
            info.text(0.01, 0.65, 'Epsilon :' + str(param_algo_descente['epsilon']), fontsize=8)
            info.text(0.01, 0.61, "Moment d'ordre " + str(param_algo_descente['moment']), fontsize=8)
        elif algo_utilise== 'Adam':
            info.text(0.01, 0.73, 'Momentum : ' +
                      str(param_algo_descente['momentum']), fontsize=8)
            info.text(0.01, 0.69, 'Gamma : ' + str(param_algo_descente['gamma']), fontsize=8)
            info.text(0.01, 0.65, 'Gamma 1 : ' + str(param_algo_descente['gamma_1']), fontsize=8)
            info.text(0.01, 0.61, 'Gamma 2 : ' + str(param_algo_descente['gamma_2']), fontsize=8)
            info.text(0.01, 0.57, 'Epsilon :' + str(param_algo_descente['epsilon']), fontsize=8)
            info.text(0.01, 0.53, "Alpha" + str(param_algo_descente['alpha']), fontsize=8)





        info.text(0.01, 0.31, "Infos courbe", fontsize=12)
        info.text(0.01, 0.27, 'Nombre de parties parallèles : ' + str(param['nb_exp']),
                  fontsize=8)
        info.text(0.01, 0.23, "Nombre d'apprentissages : " + str(param['training_size']*param['learning_set_pass_nb']//param['batch_size']),
                  fontsize=8)
        info.text(0.01, 0.19, "Taille de batch : " + str(param['batch_size']),
                  fontsize=8)
        info.text(0.01, 0.15, 'Test tous les ' + str(param['test_period']) + ' apprentissages',
                  fontsize=8)
        info.text(0.01, 0.11, 'Moyenne sur ' + str(param['testing_size']) + ' samples par test',
                  fontsize=8)

        plt.title("Evolution du succès, test effectué tous les " + str(self.test_period)
                  + " apprentissages")


        # saves the plot in directory
        save_date = strftime('%Y-%m-%d-%H%M%S', gmtime())
        namefile = str(save_date) + '.png'
        namefile = self._name + "/" + namefile
        plt.savefig(namefile,)
        plt.close()

    def plot(self, errors_during_learning, param, param_algo_descente):
        # plots the mean error as a function of time
        fig = plt.figure()

        gs = GridSpec(1, 20)

        ax_D_x = fig.add_subplot(gs[0, 0:10])
        ax_D_x.autoscale(axis='x')
        if np.shape(errors_during_learning) == (len(errors_during_learning),):
            data = errors_during_learning
            errorbar = np.zeros(len(errors_during_learning))
        else:
            data = np.mean(errors_during_learning, 0)
            std = np.std(errors_during_learning, 0)

            errorbar = std / np.sqrt(len(errors_during_learning))


        ax_D_x.set_xlabel("Nombre de parties (X" + str(param['test_period']) + ")")

        ax_D_x.legend(loc=0, bbox_to_anchor=(1.3, 0.1))
        plt.ylabel("Succès moyen sur le batch de test pour les " + str(self.learning_iterations)
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


        algo_utilise = param_algo_descente['algo_utilise']
        info.text(0.01, 0.31, 'Algorithme : ' +
                  str(algo_utilise), fontsize=8)
        if algo_utilise == 'Gradient':
            info.text(0.01, 0.35, 'Momentum : ' +
                      str(param_algo_descente['momentum']), fontsize=8)
            info.text(0.01, 0.39, 'Eta : ' +
                      str(param_algo_descente['eta']), fontsize=8)
        elif algo_utilise == 'Adagrad':
            info.text(0.01, 0.35, 'Momentum : ' +
                      str(param_algo_descente['momentum']), fontsize=8)
            info.text(0.01, 0.39, 'Eta : ' +
                      str(param_algo_descente['eta']), fontsize=8)
            info.text(0.01, 0.43, 'Epsilon : ' +
                      str(param_algo_descente['epsilon']), fontsize=8)
        elif algo_utilise == 'RMSProp':
            info.text(0.01, 0.35, 'Momentum : ' +
                      str(param_algo_descente['momentum']), fontsize=8)
            info.text(0.01, 0.39, 'Eta : ' +
                      str(param_algo_descente['eta']), fontsize=8)
            info.text(0.01, 0.43, 'Gamma : ' + str(param_algo_descente['gamma']), fontsize=8)
            info.text(0.01, 0.47, 'Epsilon : ' +
                      str(param_algo_descente['epsilon']), fontsize=8)
            info.text(0.01, 0.51, "Moment d'ordre " + str(param_algo_descente['moment']), fontsize=8)
        elif algo_utilise == 'Adadelta':
            info.text(0.01, 0.35, 'Momentum : ' +
                      str(param_algo_descente['momentum']), fontsize=8)
            info.text(0.01, 0.39, 'Gamma : ' + str(param_algo_descente['gamma']), fontsize=8)
            info.text(0.01, 0.43, 'Epsilon :' + str(param_algo_descente['epsilon']), fontsize=8)
            info.text(0.01, 0.47, "Moment d'ordre " + str(param_algo_descente['moment']), fontsize=8)
        elif algo_utilise == 'Adam':
            info.text(0.01, 0.35, 'Momentum : ' +
                      str(param_algo_descente['momentum']), fontsize=8)
            info.text(0.01, 0.39, 'Gamma : ' + str(param_algo_descente['gamma']), fontsize=8)
            info.text(0.01, 0.43, 'Gamma 1 : ' + str(param_algo_descente['gamma_1']), fontsize=8)
            info.text(0.01, 0.47, 'Gamma 2 : ' + str(param_algo_descente['gamma_2']), fontsize=8)
            info.text(0.01, 0.51, 'Epsilon :' + str(param_algo_descente['epsilon']), fontsize=8)
            info.text(0.01, 0.55, "Alpha" + str(param_algo_descente['alpha']), fontsize=8)

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
        plt.title("Evolution du succès, test effectué tous les " + str(self.test_period)
                  + " apprentissages")
