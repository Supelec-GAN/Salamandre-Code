import unittest
from brain.neuronLayer import ConvolutionalLayer
from function.activationFunction import Relu
import numpy as np


class ConvolutionalLayerTest(unittest.TestCase):

    def setUp(self):
        weights = np.array([[[[0, 0, 1],
                              [0, 1, 0],
                              [0, 1, 0]]],
                            [[[1, 0, 1],
                              [0, 1, 0],
                              [1, 0, 1]]]], dtype=np.float64)

        bias = np.array([0, 0], dtype=np.float64)

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

        self.batch_cl = ConvolutionalLayer(activation_function=fun,
                                            input_size=(5, 5),
                                            output_size=(3, 3),
                                            filter_size=(3, 3),
                                            input_feature_maps=1,
                                            output_feature_maps=2,
                                            convolution_mode='valid',
                                            learning_batch_size=2)
        self.batch_cl.restore_coefs(coefs)

        self.flat_inputs = np.arange(50, dtype=np.float64).reshape((2, 25)).T
        self.tensor_inputs = np.arange(50, dtype=np.float64).reshape((2,1,5,5))

        self.output = np.array([[[[17, 20, 23],
                                  [32, 35, 38],
                                  [47, 50, 53]],
                                 [[30, 35, 40],
                                  [55, 60, 65],
                                  [80, 85, 90]]],
                                [[[92, 95, 98],
                                  [107, 110, 113],
                                  [122, 125, 128]],
                                 [[155, 160, 165],
                                  [180, 185, 190],
                                  [205, 210, 215]]]])

        self.error = np.ones((2, 2, 3, 3), dtype=np.float64)
        self.out_influence = np.ones((2, 2, 3, 3), dtype=np.float64)
        self.in_influence = np.array([])
        self.new_weights = np.array([[[[-162, -153, -143],
                                       [-117, -107, -99],
                                       [-72, -62, -54]]],
                                     [[[-161, -153, -143],
                                       [-117, -107, -99],
                                       [-71, -63, -53]]]], dtype=np.float64)

    def test_compute(self):
        np.testing.assert_almost_equal(self.simple_cl.compute(self.flat_inputs[:, 0:1]),
                                       self.output[0:1])
        np.testing.assert_almost_equal(self.simple_cl.compute(self.tensor_inputs[0:1]),
                                       self.output[0:1])

    def test_compute_batch(self):
        np.testing.assert_almost_equal(self.batch_cl.compute(self.flat_inputs),
                                       self.output)
        np.testing.assert_almost_equal(self.batch_cl.compute(self.tensor_inputs),
                                       self.output)

    def test_derivate_error(self):
        np.testing.assert_almost_equal(self.simple_cl.derivate_error(self.error[0:1]),
                                       self.out_influence[0:1])

    def test_backprop(self):
        self.simple_cl.compute(self.tensor_inputs[0:1])
        np.testing.assert_almost_equal(self.simple_cl.backprop(self.out_influence[0:1]),
                                       self.new_weights)


if __name__ == '__main__':
    unittest.main()
