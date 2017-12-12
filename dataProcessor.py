import numpy as np


class DataProcessor:

    def __init__(self, parallel_learnings, iterations, test_period):
        self.parallel_learnings = parallel_learnings
        self.iterations = iterations
        self.test_period = test_period

    def test_data(self, batch, mean_error_during_test):
        iteration_a_laquelle_batch_test = np.array(range(len(batch) // self.test_period))*100
        moyenne_erreur_sur_le_batch_test = np.mean(mean_error_during_test, 1)
        return iteration_a_laquelle_batch_test, moyenne_erreur_sur_le_batch_test,

    def learning_data(self, mean_error_during_learning):
        iterations_effectuees = range(self.iterations // 100)
        moyenne_erreur_apprentissage = np.mean(
            mean_error_during_learning, axis=1)

        return iterations_effectuees, moyenne_erreur_apprentissage

    def error_bar(self, mean_error_during_test):
        std_batch_test = np.std(mean_error_during_test, 1)
        error_bar = 2 * std_batch_test / np.sqrt(int(self.parallel_learnings))
        return error_bar