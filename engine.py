import numpy as np


class Engine:

    def __init__(self, net, eta, learning_set, learning_fun, testing_set, testing_fun, success_fun,
                 momentum=0, nb_exp=1, test_period=100, learning_set_pass_nb=1,
                 randomize_learning_set=True):
        """
        Classe gérant l'apprentissage d'un réseau, tout en fournissant des données au fur et à
        mesure

        :param net:
        :param eta:
        :param learning_set:
        :param learning_fun:
        :param testing_set:
        :param testing_fun:
        :param success_fun:
        :param momentum:
        :param nb_exp:
        :param test_period:
        :param learning_set_pass_nb:
        :param randomize_learning_set:
        """
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
        self.momentum = momentum
        self._randomize_learning_set = randomize_learning_set
        self._permutation = np.arange(self._learning_set_size)
        self._learning_set_pass_nb = learning_set_pass_nb
        self._batch_size = net.batch_size

        # Nombre d'apprentissage successifs
        self._nb_exp = nb_exp

        # Données recueillies pendant les apprentissages et paramètres
        self._test_period = test_period
        self._test_count = (self._learning_set_size*self._learning_set_pass_nb) \
            // self._batch_size // self._test_period
        self._error_during_learning = np.zeros((self._nb_exp, self._test_count))
        self._success_fun = success_fun

    def learn(self):
        self.net.reset()
        train_success_rate = np.zeros((self._test_count))
        test_success_rate = np.zeros((self._test_count))
        for pass_nb in range(self._learning_set_pass_nb):
            print('itération n° ' + str(pass_nb))
            # Boucle pour une fois le set d'entrainement
            for batch_nb in range(self._learning_set_size // self._batch_size):
                # print("batch n° ", str(batch_nb))
                debut = batch_nb * self._batch_size
                fin = (batch_nb+1) * self._batch_size
                intervalle = self._permutation[debut:fin]
                self.net.compute(self._learning_set[intervalle])
                expected_output = self._learning_fun.label(intervalle)
                self.net.backprop(expected_output)

                # Enregistrement périodique de l'erreur sur le set de test
                if (pass_nb*self._learning_set_size + batch_nb) % self._test_period == 0:
                    test_number = (pass_nb*self._learning_set_size // self._batch_size
                                   + batch_nb) // self._test_period
                    train_success_rate[test_number] = self.get_training_success_rate()
                    test_success_rate[test_number] = self.get_testing_success_rate()
                    print("batch n° ", str(batch_nb))
                    print('Succès : ', str(train_success_rate[test_number]), ' ', str(test_success_rate[test_number]))

        return train_success_rate

    def get_current_error(self):
        """Calcule l'erreur courante du réseau sur le set de test

        :return: The mean error of the network for the testing set
        """

        self.net.batch_size = self._testing_set_size
        output = self.net.compute(self._testing_set)
        expected_output = self._testing_fun.label(np.arange(self._testing_set_size))
        error_during_testing = self.net.error.out(output, expected_output)
        mean_error = np.mean(error_during_testing)
        self.net.batch_size = self._batch_size
        return mean_error

    def get_training_success_rate(self):
        """Calcule le taux de succès du réseau sur le set d'entrainement

        :return: The success rate of the network for the training set
        """
        self.net.batch_size = self._testing_set_size
        output = self.net.compute(self._learning_set[:self._testing_set_size])
        expected_output = self._learning_fun.label(np.arange(self._testing_set_size))
        success_during_testing = self._success_fun(output, expected_output)
        success_rate = np.mean(success_during_testing)
        self.net.batch_size = self._batch_size
        return success_rate

    def get_testing_success_rate(self):
        """Calcule le taux de succès du réseau sur le set de test

        :return: The success rate of the network for the testing set
        """
        self.net.batch_size = self._testing_set_size
        output = self.net.compute(self._testing_set)
        expected_output = self._testing_fun.label(np.arange(self._testing_set_size))
        success_during_testing = self._success_fun(output, expected_output)
        success_rate = np.mean(success_during_testing)
        self.net.batch_size = self._batch_size
        return success_rate

    def run(self):
        """Effectue les n apprentissages

        :return: The error of the network during each learning cycle
        """

        for i in range(self._nb_exp):
            print("run n°", i)
            if self._randomize_learning_set:
                self._permutation = np.random.permutation(self._learning_set_size)
            self._error_during_learning[i] = self.learn()
        return self._error_during_learning
