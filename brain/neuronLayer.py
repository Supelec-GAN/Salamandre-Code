import numpy as np
from function.activationFunction import Function
from scipy.signal import convolve2d
from dataInterface import DataInterface


class Layer:

    def __init__(self, *args, **kwargs):
        """
        Defines a abstract class for a network layer
        """
        pass

    def compute(self, *args, **kwargs):
        """
        Calculate the output of the layer

        :return: Output of the layer
        """
        raise NotImplementedError

    def backprop(self, *args, **kwargs):
        """
        Backpropagates the error throught the layer

        :return: The input error
        """
        raise NotImplementedError


class ReshapeLayer(Layer):

    def __init__(self, old_shape, new_shape, *args, **kwargs):
        """
        A layer necessary between layers that use different formats of inputs.
        Shapes should be given with -1 for the batch size.

        :param old_shape: Shape of the inputs (eg. (-1, 784))
        :param new_shape: Shape of the output (eg. (-1, 1, 28, 28))
        """
        super(ReshapeLayer, self).__init__(*args, **kwargs)
        self._old_shape = old_shape
        self._new_shape = new_shape

        self._input_size = -1 * np.prod(self._old_shape)
        self._output_size = -1 * np.prod(self._new_shape)

        self.input = np.ndarray
        self.output = np.ndarray

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    def compute(self, inputs):
        self.input = inputs
        self.output = inputs.reshape(self._new_shape)
        return self.output

    def backprop(self, out_error, *args, **kwargs):
        return out_error.reshape(self._old_shape)


class NeuronLayer(Layer):

    def __init__(self, activation_function=Function(), batch_size=1,
                 descent_params_file='config_algo_descente.ini',
                 descent_params='Parametres de descente', experience_number=1, *args, **kwargs):
        super(NeuronLayer, self).__init__(*args, **kwargs)
        # The activation function is vectorized to allow batch compute
        self._activation_function = activation_function
        self._activation_function.vectorize()

        self._batch_size = batch_size

        # Weights and associated arrays are initialized as numpy ndarray
        self._weights = np.ndarray
        self._update_weights_value = np.ndarray
        self._weights_gradients_sum = np.ndarray
        self._weights_moment = np.ndarray
        self._weights_eta = np.ndarray

        # Weights and associated arrays are initilized as numpy ndarray
        self._bias = np.ndarray
        self._update_bias_value = np.ndarray
        self._bias_gradients_sum = np.ndarray
        self._bias_moment = np.ndarray
        self._bias_eta = np.ndarray

        # Reads params in file
        data_interface = DataInterface()
        conf = data_interface.read_conf(descent_params_file, descent_params)
        param_list = data_interface.extract_param(conf, experience_number)
        self.algo_utilise = param_list['algo_utilise']
        self.eta = param_list['eta']
        self.momentum = param_list['momentum']
        self.epsilon = param_list['epsilon']
        self.gamma = param_list['gamma']
        self.moment = param_list['moment']
        self.alpha = param_list['alpha']
        self.gamma_1 = param_list['gamma_1']
        self.gamma_2 = param_list['gamma_2']
        self.instant = 0

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        self._weights = new_weights

    @weights.deleter
    def weights(self):
        del self._weights

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, new_bias):
        self._bias = new_bias

    @bias.deleter
    def bias(self):
        del self._bias

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size):
        raise NotImplementedError

    def update_momentum(self, bias_influence, weight_influence):
        """

        :param bias_influence:
        :param weight_influence:
        :return: None
        """

        if self.algo_utilise == "Gradient":
            self._update_weights_value = self.momentum * self._update_weights_value - self.eta * weight_influence
            self._update_bias_value = self.momentum * self._update_bias_value + self.eta * bias_influence

        elif self.algo_utilise == "Adagrad":
            self._weights_gradients_sum = self._weights_gradients_sum + weight_influence ** 2
            partial = np.sqrt(np.add(self._weights_gradients_sum, self.epsilon))
            self._update_weights_value = self.momentum * self._update_weights_value - self.eta * np.divide(
                weight_influence, partial)

            self._bias_gradients_sum = self._bias_gradients_sum + bias_influence ** 2
            partial = np.sqrt(np.add(self._bias_gradients_sum, self.epsilon))
            self._update_bias_value = self.momentum * self._update_bias_value + self.eta * np.divide(
                bias_influence, partial)

        elif self.algo_utilise == "RMSProp":

            if self.moment == 2:
                self._weights_gradients_sum = self.gamma * self._weights_gradients_sum + (
                            1 - self.gamma) * (weight_influence ** 2)
                partial = np.sqrt(np.add(self._weights_gradients_sum, self.epsilon))
                self._update_weights_value = self.momentum * self._update_weights_value - self.eta * np.divide(
                    weight_influence, partial)

                self._bias_gradients_sum = self.gamma * self._bias_gradients_sum + (
                            1 - self.gamma) * (bias_influence ** 2)
                partial = np.sqrt(np.add(self._bias_gradients_sum, self.epsilon))
                self._update_bias_value = self.momentum * self._update_bias_value + self.eta * np.divide(
                    bias_influence, partial)

            if self.moment == 1:
                self._weights_gradients_sum = self.gamma * self._weights_gradients_sum + (
                            1 - self.gamma) * weight_influence ** 2
                self._weights_moment = self.gamma * self._weights_moment + (
                            1 - self.gamma) * weight_influence
                partial = np.sqrt(np.add((self._weights_gradients_sum - self._weights_moment ** 2),
                           self.epsilon))
                self._update_weights_value = self.momentum * self._update_weights_value - self.eta * np.divide(
                    weight_influence, partial)

                self._bias_gradients_sum = self.gamma * self._bias_gradients_sum + (
                            1 - self.gamma) * bias_influence ** 2
                self._bias_moment = self.gamma * self._bias_moment + (
                            1 - self.gamma) * bias_influence
                partial = np.sqrt(
                np.add((self._bias_gradients_sum - self._bias_moment ** 2), self.epsilon))
                self._update_bias_value = self.momentum * self._update_bias_value + self.eta * np.divide(
                    bias_influence, partial)

        elif self.algo_utilise == "Adadelta":

            if self.moment == 2:
                self._weights_gradients_sum = self.gamma * self._weights_gradients_sum + (
                            1 - self.gamma) * weight_influence ** 2
                self._weights_eta = self.gamma * self._weights_eta + (
                            1 - self.gamma) * self._update_weights_value ** 2
                partial = np.sqrt(np.add(self._weights_eta, self.epsilon)) * weight_influence
                partial2 = np.sqrt(np.add(self._weights_gradients_sum, self.epsilon))
                self._update_weights_value = self.momentum * self._update_weights_value - np.divide(
                    partial, partial2)

                self._bias_gradients_sum = self.gamma * self._bias_gradients_sum + (
                            1 - self.gamma) * bias_influence ** 2
                self._bias_eta = self.gamma * self._bias_eta + (
                            1 - self.gamma) * self._update_bias_value ** 2
                partial = np.sqrt(np.add(self._bias_eta, self.epsilon)) * bias_influence
                partial2 = np.sqrt(np.add(self._bias_gradients_sum, self.epsilon))
                self._update_bias_value = self.momentum * self._update_bias_value + np.divide(
                    partial, partial2)

            if self.moment == 1:
                self._weights_gradients_sum = self.gamma * self._weights_gradients_sum + (
                            1 - self.gamma) * weight_influence ** 2
                self._weights_eta = self.gamma * self._weights_eta + (
                            1 - self.gamma) * self._update_weights_value ** 2
                self._weights_moment = self.gamma * self._weights_moment + (
                            1 - self.gamma) * weight_influence
                partial = np.sqrt(np.add(self._bias_eta, self.epsilon)) * bias_influence
                partial2 = np.sqrt(
                    np.add((self._weights_gradients_sum - self._weights_moment ** 2),
                           self.epsilon))
                self._update_weights_value = self.momentum * self._update_weights_value - np.divide(
                    partial, partial2)

                self._bias_gradients_sum = self.gamma * self._bias_gradients_sum + (
                            1 - self.gamma) * bias_influence ** 2
                self._bias_eta = self.gamma * self._bias_eta + (
                            1 - self.gamma) * self._update_bias_value ** 2
                self._bias_moment = self.gamma * self._bias_moment + (
                            1 - self.gamma) * bias_influence
                partial = np.sqrt(np.add(self._bias_eta, self.epsilon)) * bias_influence
                partial2 = np.sqrt(
                    np.add((self._bias_gradients_sum - self._bias_moment ** 2), self.epsilon))
                self._update_bias_value = self.momentum * self._update_bias_value + np.divide(
                    partial, partial2)

        elif self.algo_utilise == "Adam":

            self.instant += 1

            self._weights_gradients_sum = self.gamma * self._weights_gradients_sum + (
                     1 - self.gamma) * weight_influence ** 2
            self._weights_moment = self.gamma * self._weights_moment + (
                     1 - self.gamma) * weight_influence
            partial = (1 - self.gamma_1 ** self.instant)
            partial2 = np.sqrt(np.add(
                np.divide(self._weights_gradients_sum, (1 - self.gamma_2 ** self.instant)),
                self.epsilon))
            self._update_weights_value = self.momentum * self._update_weights_value - self.alpha * \
                                         np.divide(np.divide(self._weights_moment, partial), partial2)

            self._bias_gradients_sum = self.gamma * self._bias_gradients_sum + (
                        1 - self.gamma) * bias_influence ** 2
            self._bias_moment = self.gamma * self._bias_moment + (
                        1 - self.gamma) * bias_influence
            partial = (1 - self.gamma_1 ** self.instant)
            partial2 = np.sqrt(
                np.add(np.divide(self._bias_gradients_sum, (1 - self.gamma_2 ** self.instant)),
                       self.epsilon))
            self._update_bias_value = self.momentum * self._update_bias_value + self.alpha * \
                                      np.divide(np.divide(self._bias_moment, partial), partial2)

    def update_weights(self):
        """
        Updates weights according to update_weights_value that was calculated previously

        :return: None
        """
        self._weights = self._weights + self._update_weights_value

    def update_bias(self):
        """
        Updates bias according to update_bias_value that was calculated previously

        :return: None
        """
        self._bias = self._bias + self._update_bias_value

    def calculate_weight_influence(self, out_influence):
        """
        Calculates the weights influence

        :param out_influence: Influence of output on the error
        :return: Array of the same dimension than weights
        """
        raise NotImplementedError

    def calculate_bias_influence(self, out_influence):
        """
        Calculates the bias influence

        :param out_influence: Influence of output on the error
        :return: Array of the same dimension than bias
        """
        raise NotImplementedError

    def derivate_error(self, in_influence):
        """
        Calculates the error derivation

        :param in_influence: Influence of the input of the next layer
        :return: Error used in the recursive formula
        """
        raise NotImplementedError

    def input_error(self, out_influence, new_weights):
        """
        Propagates the error from the activation levels to the inputs

        :param out_influence: Influence of the activation levels
        :param new_weights: The updated weights
        :return: Influence of the input
        """
        raise NotImplementedError

    def save_coefs(self):
        """
        Creates a dictionary with the current state of the layer

        :return: A dictionary of weights and bias
        """
        coefs = {'weights': self._weights,
                 'bias': self._bias}
        return coefs

    def restore_coefs(self, coefs):
        """
        Can restore a layer from a dictionary created by save_coefs

        :param coefs: A dictionary with weights and bias
        :return: None
        """
        self._weights = coefs['weights']
        self._bias = coefs['bias']


class FullyConnectedLayer(NeuronLayer):

    def __init__(self, input_size=1, output_size=1, noise_size=0, *args, **kwargs):
        """
        Creates a fully connected neuron layer

        :param activation_function:
        :param input_size:
        :param output_size:
        :param noise_size:
        :param batch_size:
        :param param_desc:
        :param nb_exp:
        """
        super(FullyConnectedLayer, self).__init__(*args, **kwargs)
        self._input_size = input_size
        self._output_size = output_size
        self._noise_size = noise_size

        self.input = np.zeros((self._batch_size, self._input_size))
        self.noise_input = np.zeros((self._batch_size, self._noise_size))
        self.activation_levels = np.zeros((self._batch_size, self._output_size))
        self.output = np.zeros((self._batch_size, self.output_size))

        self._weights = np.random.randn(self._input_size+self._noise_size, self._output_size)
        self._update_weights_value = np.zeros_like(self._weights)
        self._weights_gradients_sum = np.zeros_like(self._weights)
        self._weights_moment = np.zeros_like(self._weights)
        self._weights_eta = np.zeros_like(self._weights)

        self._bias = np.zeros(self._output_size)
        self._update_bias_value = np.zeros_like(self._bias)
        self._bias_gradients_sum = np.zeros_like(self._bias)
        self._bias_moment = np.zeros_like(self._bias)
        self._bias_eta = np.zeros_like(self._bias)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size):
        self.input = np.zeros((new_batch_size, self._input_size))
        self.activation_levels = np.zeros((new_batch_size, self._output_size))
        self.output = np.zeros((new_batch_size, self._output_size))
        self.noise_input = np.zeros((new_batch_size, self._noise_size))
        self._batch_size = new_batch_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    def compute(self, inputs):
        self.input = inputs
        if self._noise_size != 0:  # nécessaire car np.zeros((0,1)) est un objet chelou
            self.noise_input = np.random.randn(self._batch_size, self._batch_size)
            self.input = np.hstack((self.input, self.noise_input))
        self.activation_levels = np.dot(self.input, self._weights) - self._bias
        self.output = self._activation_function.out(self.activation_levels)
        return self.output

    def backprop(self, out_error, update=True):
        out_influence = self.derivate_error(out_error)
        weight_influence = self.calculate_weight_influence(out_influence)
        bias_influence = self.calculate_bias_influence(out_influence)
        self.update_momentum(bias_influence, weight_influence)
        if update:
            self.update_weights()
            self.update_bias()
        in_error = self.input_error(out_influence, updated=update)
        return in_error

    def calculate_weight_influence(self, out_influence):
        return np.dot(np.transpose(self.input), out_influence) / self._batch_size

    def calculate_bias_influence(self, out_influence):
        return np.mean(out_influence, axis=0)

    def derivate_error(self, out_error):
        deriv_vector = self._activation_function.derivate(self.activation_levels)
        return deriv_vector * out_error

    def input_error(self, out_influence, updated):
        if updated:
            return np.dot(out_influence,
                          np.transpose(
                              self._weights[:, :self._input_size]
                              )
                          )
        else:
            return np.dot(out_influence,
                          np.transpose(
                              (self._weights + self._update_weights_value)[:, :self._input_size]
                              )
                          )


class ConvolutionalLayer(NeuronLayer):

    def __init__(self, input_size=(1, 1), output_size=(1, 1), filter_size=(1, 1),
                 input_feature_maps=1, output_feature_maps=1, convolution_mode='valid', step=1,
                 *args, **kwargs):
        """
        Creates a convolutional layer for a neural network

        :param input_size:
        :param output_size:
        :param filter_size:
        :param input_feature_maps:
        :param output_feature_maps:
        :param convolution_mode:
        :param step:
        """
        super(ConvolutionalLayer, self).__init__(*args, **kwargs)
        self._input_size = input_size
        self._output_size = output_size
        self._filter_size = filter_size
        self._input_feature_maps = input_feature_maps
        self._output_feature_maps = output_feature_maps
        self._step = step  # Laisser à 1 pour l'instant
        self._convolution_mode = convolution_mode

        self._weights = np.random.randn(self._output_feature_maps, self._input_feature_maps,
                                        self._filter_size[0], self._filter_size[1])
        self._update_weights_value = np.zeros_like(self._weights)
        self._weights_gradients_sum = np.zeros_like(self._weights)
        self._weights_moment = np.zeros_like(self._weights)
        self._weights_eta = np.zeros_like(self._weights)

        self._bias = np.zeros(self._output_feature_maps)
        self._update_bias_value = np.zeros_like(self._bias)
        self._bias_gradients_sum = np.zeros_like(self._bias)
        self._bias_moment = np.zeros_like(self._bias)
        self._bias_eta = np.zeros_like(self._bias)

        self._input_size = input_size
        self._output_size = output_size
        if self._convolution_mode == 'full':
            self._output_size = (self._input_size[0] + (self._filter_size[0]-1),
                                 self._input_size[1] + (self._filter_size[1]-1))
            self._reverse_convolution_mode = 'valid'
        elif self._convolution_mode == 'same':
            self._output_size = self._input_size
            self._reverse_convolution_mode = 'same'
        elif self._convolution_mode == 'valid':
            self._output_size = (self._input_size[0] - (self._filter_size[0]-1),
                                 self._input_size[1] - (self._filter_size[1]-1))
            self._reverse_convolution_mode = 'full'
        else:
            raise Exception("Invalid convolution mode")
        self.input = np.zeros((self._batch_size, self._input_feature_maps,
                               self._input_size[0], self._input_size[1]))
        self.activation_levels = np.zeros((self._batch_size, self._output_feature_maps,
                                           self._output_size[0], self._output_size[1]))
        self.output = np.zeros((self._batch_size, self._output_feature_maps,
                                self._output_size[0], self._output_size[1]))

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size):
        self.input = np.zeros((new_batch_size, self._input_feature_maps,
                               self._input_size[0], self._input_size[0]))
        self.activation_levels = np.zeros((new_batch_size, self._output_feature_maps,
                                           self._output_size[0], self._output_size[1]))
        self.output = np.zeros((new_batch_size, self._output_feature_maps,
                                self._output_size[0], self._output_size[1]))
        self._batch_size = new_batch_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    def compute(self, inputs):
        self.input = inputs
        self.activation_levels = self.conv2d() + self._bias[np.newaxis, :, np.newaxis, np.newaxis]
        self.output = self._activation_function.out(self.activation_levels)
        return self.output

    def backprop(self, out_error, update=True):
        out_influence = self.derivate_error(out_error)
        weight_influence = self.calculate_weight_influence(out_influence)
        bias_influence = self.calculate_bias_influence(out_influence)
        self.update_momentum(bias_influence, weight_influence)
        if update:
            self.update_weights()
            self.update_bias()
        in_error = self.input_error(out_influence, updated=update)
        return in_error

    def calculate_weight_influence(self, out_influence):
        return self.weights_conv2d(out_influence) / self._batch_size

    def calculate_bias_influence(self, out_influence):
        return np.mean(out_influence, axis=(0, 2, 3))

    def derivate_error(self, in_influence):
        deriv_vector = self._activation_function.derivate(self.activation_levels)
        return deriv_vector * in_influence

    def input_error(self, out_influence, updated):
        if updated:
            return self.reverse_conv2d(out_influence, self._weights)
        else:
            return self.reverse_conv2d(out_influence, self._weights + self._update_weights_value)

    def conv2d(self):
        out = np.zeros_like(self.output)
        for b in range(self._batch_size):
            for o in range(self._output_feature_maps):
                for i in range(self._input_feature_maps):
                    conv = convolve2d(self.input[b][i],
                                      self._weights[o][i],
                                      mode=self._convolution_mode)
                    out[b][o] += conv[::self._step, ::self._step]
        return out

    def reverse_conv2d(self, out_influence, new_weights):
        in_influence = np.zeros_like(self.input)
        for b in range(self._batch_size):
            for i in range(self._input_feature_maps):
                for o in range(self._output_feature_maps):
                    conv = convolve2d(out_influence[b][o],
                                      np.rot90(new_weights[o][i], k=2),
                                      mode=self._reverse_convolution_mode)
                    in_influence[b][i] += conv
        return in_influence

    def weights_conv2d(self, out_influence):
        weight_influence = np.zeros_like(self._weights)
        for o in range(self._output_feature_maps):
            for i in range(self._input_feature_maps):
                for b in range(self._batch_size):
                    conv = convolve2d(out_influence[b][o],
                                      np.rot90(self.input[b][i], k=2),
                                      mode=self._convolution_mode)
                    weight_influence[o][i] += conv
        return weight_influence


class MaxPoolingLayer(NeuronLayer):

    def __init__(self, input_size=(1, 1), output_size=(1, 1), pooling_size=(1, 1), feature_maps=1,
                 *args, **kwargs):
        """
        Creates a Max Pooling layer for neural network

        :param input_size: Size of the input images
        :param output_size: Size of the output images
        :param pooling_size: Size of the downscaling filter
        :param feature_maps: Number of feature maps from the previous convolutional layer
        """
        super(MaxPoolingLayer, self).__init__(*args, **kwargs)
        self._input_size = input_size
        self._output_size = output_size
        self._pooling_size = pooling_size
        self._feature_maps = feature_maps
        self._weights = np.ones((self._feature_maps, self._input_size[0], self._input_size[1]))
        self._bias = 0
        self.input = np.zeros((self._batch_size, self._feature_maps,
                               self._input_size[0], self._input_size[1]))
        self.output = np.zeros((self._batch_size, self._feature_maps,
                                self._output_size[0], self._output_size[1]))

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size):
        self.input = np.zeros((new_batch_size, self._feature_maps,
                               self._input_size[0], self._input_size[0]))
        self.output = np.zeros((new_batch_size, self._feature_maps,
                                self._output_size[0], self._output_size[1]))
        self._batch_size = new_batch_size

    def compute(self, inputs):
        """
        Compute for maxpooling layers. It is a special compute that also modifies the weights of the
        layer for the backprop

        :param inputs: The inputs (a batch of flat inputs, or a tensor with a good shape)
        :return: None
        """
        self.input = inputs
        for b in range(self._batch_size):
            for f in range(self._feature_maps):
                for h in range(0, self._input_size[0], self._pooling_size[0]):
                    for w in range(0, self._input_size[1], self._pooling_size[1]):
                        part = self.input[b][f][h:h + self._pooling_size[0],
                                                w:w + self._pooling_size[1]]
                        maxi = np.amax(part)
                        weight_part = np.zeros_like(part)
                        weight_part[part == maxi] = 1
                        self.output[b][f][h//self._pooling_size[0]][w//self._pooling_size[1]] = maxi
                        self._weights[f][h:h+self._pooling_size[0], w:w+self._pooling_size[1]] = \
                            weight_part

    def backprop(self, out_error, *args):
        """
        Backprop of a maxpooling layer. It does basically nothing, it just returns the weights of
        the layer to be compatible with the other types of layers

        :param out_error: Error of the layer
        :return: The layer's weights
        """
        out_influence = self.derivate_error(out_error)
        in_error = self.input_error(out_influence)
        return in_error

    def derivate_error(self, out_error):
        """
        There is no activation levels here, so no error to derivate.

        :param out_error: The error of the next layer
        :return: The error of the next layer
        """
        return out_error

    def input_error(self, out_influence, *args):
        """
        The error is propagated throught the maxpooling layer for the previous layer. The error is
        upscaled, then multiply element-wise with the weights created by the compute

        :param out_influence: The error to propagate
        :return: The propagated error that can be fed to the previous layer
        """
        return np.kron(out_influence, np.ones(self._pooling_size)) * self._weights


class ClippedNeuronLayer(FullyConnectedLayer):

    def __init__(self, clipping=0, *args, **kwargs):
        """
        Creates a Fully Connected layer with clipped weights for a neural network

        :param activation_function:
        :param input_size:
        :param output_size:
        :param batch_size:
        :param clipping
        """
        super(ClippedNeuronLayer, self).__init__(*args, **kwargs)
        self._clipping = clipping

    def update_weights(self):
        self._weights = self._weights + self._update_weights_value
        self.weights_clipping()

    def update_bias(self):
        self._bias = self._bias + self._update_bias_value
        self.bias_clipping()

    def weights_clipping(self):
        """
        Clips the weight in [-clipping, +clipping] with linear transformations.

        :return: None
        """
        max_weigth = np.amax(np.abs(self._weights))
        self._weights = self._clipping*self._weights/max_weigth

    def bias_clipping(self):
        """
        Clips the bias in [-clipping, +clipping] with linear transformations.

        :return: None
        """
        max_bias = np.amax(np.abs(self._bias))
        self._bias = self._clipping*self._bias/max_bias
