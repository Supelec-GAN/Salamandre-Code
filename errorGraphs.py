import matplotlib.pyplot as plt
import numpy as np
import os
from time import gmtime, strftime


class ErrorGraphs:

    def __init__(self, name, learning_iterations, eta, net, test_period):
        self._name = name
        self.learning_iterations = learning_iterations
        self.eta = eta
        self.net = net
        self.test_period = test_period

    def save(self, errors_during_learning):
        # create directory if it doesn't exist
        if not os.path.exists(self._name):
            os.mkdir(self._name)
        # plots the mean error as a function of time
        if (np.shape(errors_during_learning) == (len(errors_during_learning),)):
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
                     yerr=np.ones(len(errorbar)),
                     capthick = 1,
                     ecolor = 'k',
                     )
        plt.ylabel("Erreur moyenne sur le batch de test pour les " + str(self.learning_iterations) + " runs")
        plt.xlabel("Apprentissages")
        plt.title("Evolution de l'erreur, test effectué tous les " + str(self.test_period) + " apprentissages")
        plt.suptitle("eta =" + str(self.eta) + ";" + "Réseau en " + str(self.net))
        # saves the plot in directory
        save_date = strftime('%Y-%m-%d-%H%M%S', gmtime())
        namefile = str(save_date) + '.png'
        plt.savefig(namefile,)

    def plot(self, errors_during_learning):
        # plots the mean error as a function of time
        if (np.shape(errors_during_learning) == (len(errors_during_learning),)):
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
                     yerr=np.ones(len(errorbar)),
                     capthick = 1,
                     ecolor = 'k',
                     )
        plt.ylabel("Erreur moyenne sur le batch de test pour les " + str(self.learning_iterations) + " runs")
        plt.xlabel("Apprentissages")
        plt.title("Evolution de l'erreur, test effectué tous les " + str(self.test_period) + " apprentissages")
        plt.suptitle("eta =" + str(self.eta) + "\n" + "Réseau en " + str(self.net))
        # saves the plot in directory
        plt.show()
