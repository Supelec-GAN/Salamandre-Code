import matplotlib.pyplot as plt


class ErrorGraphs:

    def __init__(self, parallel_learnings, error_bar, eta, neuron_count, test_period):
        self.parallel_learnings = parallel_learnings
        self.error_bar = error_bar
        self.eta = eta
        self.neuron_count = neuron_count
        self.test_period = test_period

    def error_graphs_test(self, abs_error_test, ord_error_test):
        plt.figure()
        plt.errorbar(abs_error_test, ord_error_test, self.error_bar, None, fmt='x', ecolor='k', capthick=2)
        plt.ylabel("Erreur moyenne sur le batch de test pour les " +
                   str(self.parallel_learnings) + " runs")
        plt.xlabel("Apprentissages")
        plt.title("Evolution de l'erreur, test effectué tous les " +
                  str(self.test_period) + " apprentissages")
        plt.suptitle("eta =" + str(self.eta) + "\n" + "Réseau en " + str(self.neuron_count[1:]))
        plt.show()
