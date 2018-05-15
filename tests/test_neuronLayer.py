import unittest
from brain.neuronLayer import NeuronLayer
from function.activationFunction import Sigmoid
import numpy as np


class NeuronLayerTest(unittest.TestCase):

    def setUp(self):
        weights = np.array([[0.1, 0.5, 0.4, -0.8, 0.5, -0.3, 0.7, 1, -0.6, 0.1],
                            [-0.5, 0.8, 0.1, -0.2, -0.6, 0.4, 0.3, 0.9, -1, 1.3],
                            [-1.1, -0.2, -0.6, -0.5, 0.7, 0.7, 0.1, -0.1, 1, 0.1]])
        bias = np.array([[-1.1],
                         [0.2],
                         [0.8]])
        coefs = {'weights': weights,
                 'bias': bias}

        fun = Sigmoid(0.1)
        fun.vectorize()

        self.simple_nl = NeuronLayer(activation_function=fun,
                                     input_size=10,
                                     output_size=3,
                                     noise_size=0,
                                     learning_batch_size=1,
                                     param_desc='Parametres de descente',
                                     nb_exp=0)
        self.simple_nl.restore_coefs(coefs)

        self.batch_nl = NeuronLayer(activation_function=fun,
                                    input_size=10,
                                    output_size=3,
                                    noise_size=0,
                                    learning_batch_size=3,
                                    param_desc='Parametres de descente',
                                    nb_exp=0)
        self.batch_nl.restore_coefs(coefs)

        self.inputs = np.array([[0.1, 0.15, -1],
                                [0.2, 0.25, -0.9],
                                [0.3, 0.35, -0.8],
                                [0.4, 0.45, -0.7],
                                [0.5, 0.55, -0.6],
                                [0.6, 0.65, -0.5],
                                [0.7, 0.75, -0.4],
                                [0.8, 0.85, -0.3],
                                [0.9, 0.95, -0.2],
                                [1, 0.05, -0.1]])

        self.expected_output = np.array([[0.54810078, 0.54760536, 0.50424990],
                                         [0.52821998, 0.49762502, 0.48700293],
                                         [0.51074834, 0.50837422, 0.50799932]])

    def test_compute(self):
        self.assertTrue(np.allclose(self.simple_nl.compute(self.inputs[:,0:1]),
                                    self.expected_output[:,0:1],
                                    rtol=1e-04,
                                    atol=1e-07))

    def test_compute_batch(self):
        self.assertTrue(np.allclose(self.batch_nl.compute(self.inputs),
                                    self.expected_output,
                                    rtol=1e-04,
                                    atol=1e-07))


if __name__ == '__main__':
    unittest.main()
