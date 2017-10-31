import numpy as np
from xor.network import Network
from fonction import Sigmoid


class Engine:
    """
    Classe gérant l'apprentissage d'un réseau, tout en fournissant des données au fur et à mesure
    """

    def __init__(self, net, eta, learning_set, learning_fun, testing_set, testing_fun,
                 learning_iterations=1, test_period=100):
        # Réseau utilisé
        self._net = net

        # Paramètres pour l'apprentissage
        self._eta = eta

        # Données d'apprentissage (objets + étiquettes/fonction d'étiquettage)
        self._learning_set = learning_set       # données normalisées
        self._learning_fun = learning_fun
        self._learning_set_size = np.size(learning_set, 0)

        # Données de test (objets + étiquettes/fonction d'étiquettage)
        self._testing_set = testing_set         # données normalisées
        self._testing_fun = testing_fun
        self._testing_set_size = np.size(testing_set, 0)

        # Nombre d'apprentissage successifs
        self._learning_iterations = learning_iterations

        # Données recueillies pendant les apprentissages et paramètres
        self._test_period = test_period
        self._test_count = self._learning_set_size // self._test_period
        self._error_during_learning = np.zeros((self._learning_iterations, self._test_count))

    def learn(self):
        self._net.reset()
        learning_error = np.zeros(self._test_count)
        for data_nb in range(self._learning_set_size):
            self._net.compute(self._learning_set[data_nb])
            expected_output = self._learning_fun(data_nb)
            self._net.backprop(self._eta, self._learning_set[data_nb], expected_output)

            # Enregistrement périodique de  l'erreur sur le set de test
            if data_nb % self._test_period == 0:
                test_number = data_nb // self._test_period
                learning_error[test_number] = self.get_current_error()
        return learning_error

    def get_current_error(self):
        """
        Calcule l'erreur courante du réseau sur le set de test
        """
        error_during_testing = np.zeros(self._testing_set_size)
        for test_nb in range(self._testing_set_size):
            output = self._net.compute(self._testing_set[test_nb])
            expected_output = self._testing_fun(test_nb)
            error_during_testing[test_nb] = self._net.error(output, expected_output)
        mean_error = np.mean(error_during_testing)
        return mean_error

    def run(self):
        """
        Effectue les n apprentissages
        """
        for i in range(self._learning_iterations):
            self._error_during_learning[i] = self.learn()
        return self._error_during_learning
