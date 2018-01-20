import numpy as np


class Engine:
    """Classe gérant l'apprentissage d'un réseau, tout en fournissant des données au fur et à mesure
    """

    def __init__(self, net, eta, learning_set, learning_fun, testing_set, testing_fun, success_fun,
                 learning_iterations=1, test_period=100, learning_set_pass_nb=1,
                 randomize_learning_set=True):
        # Réseau utilisé
        self.net = net

        # Données d'apprentissage (objets + étiquettes/fonction d'étiquettage)
        self._learning_set = learning_set       # données normalisées
        self._learning_fun = learning_fun
        self._learning_set_size = np.size(learning_set, 0)

        # Données de test (objets + étiquettes/fonction d'étiquettage)
        self._testing_set = testing_set         # données normalisées
        self._testing_fun = testing_fun
        self._testing_set_size = np.size(testing_set, 0)

        # Paramètres pour l'apprentissage
        self.eta = eta
        self._randomize_learning_set = randomize_learning_set
        self._permutation = np.arange(self._learning_set_size)
        self._learning_set_pass_nb = learning_set_pass_nb

        # Nombre d'apprentissage successifs
        self._learning_iterations = learning_iterations

        # Données recueillies pendant les apprentissages et paramètres
        self._test_period = test_period
        self._test_count = (self._learning_set_size*self._learning_set_pass_nb) // self._test_period
        self._error_during_learning = np.zeros((self._learning_iterations, self._test_count))
        self._success_fun = success_fun

    def learn(self):
        self.net.reset()
        testing_success_rate = np.zeros(self._test_count)
        for pass_nb in range(self._learning_set_pass_nb):
            # Boucle pour une fois le set d'entrainement
            for data_nb in range(self._learning_set_size):
                self.net.compute(self._learning_set[self._permutation[data_nb]])
                expected_output = self._learning_fun.out(self._permutation[data_nb])
                self.net.backprop(self.eta, self._learning_set[self._permutation[data_nb]],
                                  expected_output)

                # Enregistrement périodique de  l'erreur sur le set de test
                if (pass_nb*self._learning_set_size + data_nb) % self._test_period == 0:
                    test_number = (pass_nb*self._learning_set_size + data_nb) // self._test_period
                    testing_success_rate[test_number] = self.get_current_success_rate()

        return testing_success_rate

    def get_current_error(self):
        """Calcule l'erreur courante du réseau sur le set de test

        :return: The mean error of the network for the testing set
        """

        error_during_testing = np.zeros(self._testing_set_size)
        for test_nb in range(self._testing_set_size):
            output = self.net.compute(self._testing_set[test_nb])
            expected_output = self._testing_fun.out(test_nb)
            error_during_testing[test_nb] = self.net.error.out(output, expected_output)
        mean_error = np.mean(error_during_testing)
        return mean_error

    def get_current_success_rate(self):
        """Calcule le taux de succès courant du réseau sur le set de test"""
        success_during_testing = np.zeros(self._testing_set_size)
        for test_nb in range(self._testing_set_size):
            output = self.net.compute(self._testing_set[test_nb])
            expected_output = self._testing_fun.out(test_nb)
            success_during_testing[test_nb] = self._success_fun(output, expected_output)
        success_rate = np.mean(success_during_testing)
        return success_rate

    def run(self):
        """Effectue les n apprentissages

        :return: The error of the network during each learning cycle
        """

        for i in range(self._learning_iterations):
            if self._randomize_learning_set:
                self._permutation = np.random.permutation(self._learning_set_size)
            self._error_during_learning[i] = self.learn()
        return self._error_during_learning
