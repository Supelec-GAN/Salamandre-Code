import unittest
from brain.neuronLayer import ConvolutionalLayer
from function.activationFunction import Relu
import numpy as np


class ConvolutionalLayerTest(unittest.TestCase):

    def setUp(self):
        weights = np.array([[[[0, 0, 1],
                              [0, 1, 0],
                              [1, 0, 0]]],
                            [[[1, 0, 1],
                              [0, 1, 0],
                              [1, 0, 1]]]])

        bias = np.array([0, 0])

        coefs = {'weights': weights,
                 'bias': bias}

        fun = Relu()
        fun.vectorize()

        self.simple_cl = ConvolutionalLayer(activation_function=fun,
                                            input_size=(5, 5),
                                            output_size=(3, 3),
                                            filter_size=(3, 3),
                                            input_feature_maps=1,
                                            output_feature_maps=2,
                                            convolution_mode='valid',
                                            learning_batch_size=1)
        self.simple_cl.restore_coefs(coefs)

        self.flat_inputs = np.arange(25).reshape(((25, 1)))
        self.tensor_inputs = np.arange(25).reshape((1,1,5,5))

    def test_compute(self):
        expected_output = [[[[18, 21, 24],
                             [33, 36, 39],
                             [48, 51, 54]],
                            [[30, 35, 40],
                             [55, 60, 65],
                             [80, 85, 90]]]]
        self.assertTrue(np.allclose(self.simple_cl.compute(self.flat_inputs), expected_output))
        self.assertTrue(np.allclose(self.simple_cl.compute(self.tensor_inputs), expected_output))


if __name__ == '__main__':
    unittest.main()
