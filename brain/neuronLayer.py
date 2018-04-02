import numpy as np
from fonction import Function
from theano.tensor.nnet import conv2d, conv2d_transpose


class NeuronLayer:
    """Classe permettant de créer une couche de neurones"""

    def __init__(self, activation_function=Function(), input_size=1, output_size=1,
                 learning_batch_size=1):
        # Matrice de dimension q*p avec le nombre de sortie et p le nombre d'entrée
        self._input_size = input_size
        self._output_size = output_size
        self._learning_batch_size = learning_batch_size
        self._weights = np.transpose(np.random.randn(input_size, output_size))
        self._bias = np.zeros((output_size, 1))                                # Vecteur colonne
        # On peut laisser le biais comme un vecteur colonne, car en faire une matrice contenant
        # learning_batch_size fois la même colonne. Lorsque l'on aura besoin du biais dans les
        # calculs, il y aura mathématiquement parlant un problème de dimension (addition vecteur
        # + matrice), cependant numpy gère ça en additionnant le vecteur de biais à chacune des
        # colonnes de la matrice (broadcast)
        self.input = np.zeros((input_size, learning_batch_size))
        self._activation_function = activation_function
        self.activation_levels = np.zeros((output_size, learning_batch_size))  # Chaque colonne
        # correspond à une entrée du batch
        self.output = np.zeros((output_size, learning_batch_size))             # Chaque colonne
        # correspond à une entrée du batch

    @property
    def weights(self):
        """Get the current weights."""
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        self._weights = new_weights

    @weights.deleter
    def weights(self):
        del self._weights

    @property
    def bias(self):
        """Get the current bias."""
        return self._bias

    @bias.setter
    def bias(self, new_bias):
        self._bias = new_bias

    @bias.deleter
    def bias(self):
        del self._bias

    @property
    def learning_batch_size(self):
        return self._learning_batch_size

    @learning_batch_size.setter
    def learning_batch_size(self, new_learning_batch_size):
        self.activation_levels = np.zeros((self._output_size, new_learning_batch_size))
        self.output = np.zeros((self._output_size, new_learning_batch_size))
        self._learning_batch_size = new_learning_batch_size

    @property
    def output_size(self):
        return self._output_size

    ##
    # @brief      Calcul des sorties de la couche
    #
    # @param      inputs  Inputs

    def compute(self, inputs):
        self.input = self.flatten_inputs(inputs)
        self.activation_levels = np.dot(self._weights, self.input) - self._bias
        self.output = self._activation_function.out(self.activation_levels)
        return self.output

    ##
    # @brief      Retropropagation au niveau d'une couche
    #
    # @param      out_influence  influence of output on the error
    # @param      eta            The eta
    # @param      input_layer    The input value of the layer
    #
    # @return     retourne influence of the input on the error
    #
    def backprop(self, out_influence, eta, update=True):
        weight_influence = self.calculate_weight_influence(out_influence)
        bias_influence = self.calculate_bias_influence(out_influence)
        if update:
            self.update_weights(eta, weight_influence)
            self.update_bias(eta, bias_influence)
            return self._weights
        else:
            return self._weights - eta * weight_influence

    def update_weights(self, eta, weight_influence):
        self._weights = self._weights - eta * weight_influence

    def update_bias(self, eta, bias_influence):
        self._bias = self._bias + eta * bias_influence

    ##
    # @brief      Calculates the weight influence.
    #
    # @param      input_layer    input of the last compute
    # @param      out_influence  influence of output on the error
    #
    # @return     vecteur of same dimension than weights.
    #
    def calculate_weight_influence(self, out_influence):
        return np.dot(out_influence, np.transpose(self.input)) / self._learning_batch_size

    ##
    # @brief      Calculates the bias influence (which is out_influence)
    ##
    def calculate_bias_influence(self, out_influence):
        mean_out_influence = np.mean(out_influence, axis=1, keepdims=True)
        return mean_out_influence

    ##
    # @brief      Calculates the error derivation
    #
    # @param      out_influence  influence of output on the error
    #
    # @return     the error used in the recursive formula
    #
    def derivate_error(self, in_influence):
        deriv_vector = self._activation_function.derivate(self.activation_levels)
        return deriv_vector * in_influence

    def input_error(self, out_influence):
        in_influence = np.dot(np.transpose(self.weights), out_influence)
        return in_influence

    def flatten_inputs(self, inputs):
        ndim = inputs.ndim
        if ndim == 2:
            return inputs
        elif ndim == 4:
            # Maybe add a check
            inputs_reshaped = inputs.ravel().reshape((self._learning_batch_size,
                                                      self._input_size)).T
            return inputs_reshaped
        else:
            raise Exception('Wrong inputs dimension : it should be a matrix or a 4D tensor')


##
# @brief      Class for layer with noisy input added to inputs.
##
class NoisyLayer(NeuronLayer):
    def __init__(self, activation_function, input_size=1, output_size=1,
                 learning_batch_size=1, noise_size=0):
        super(NoisyLayer, self).__init__(activation_function, input_size,
                                         output_size, learning_batch_size)
        self._noise_size = noise_size
        self.weights = np.transpose(np.random.randn(input_size+noise_size, output_size))
        self.noise_input = np.zeros((noise_size, learning_batch_size))
        self._full_input = np.concatenate([self.input, self.noise_input])

    ##
    # Compute très légèrement différent, on concatène un vecteur de bruits à l'input si nécessaire
    ##
    def compute(self, inputs):
        self.input = inputs
        if self._noise_size != 0:  # nécessaire car np.zeros( (0,1)) est un objet chelou
            self.noise_input = np.random.randn(self._noise_size, self._learning_batch_size)
            self._full_input = np.concatenate([self.input, self.noise_input])
        self.activation_levels = np.dot(self.weights, inputs) - self._bias
        self.output = self._activation_function.out(self.activation_levels)
        return self.output

    ##
    # backprop très légèrement différente, on rétropropage en considérant le vecteur bruit,
    # mais sans renvoyer son influence à la couche précédente
    ##
    def backprop(self, out_influence, eta, update=True):
        weight_influence = self.calculate_weight_influence(out_influence)
        bias_influence = self.calculate_bias_influence(out_influence)
        if update:
            self.update_weights(eta, weight_influence)
            self.update_bias(eta, bias_influence)
            return self.weights[:, 0:self._input_size]  # On extrait les poids concernant les
            # vrais inputs (le bruit n'a pas besoin d'influer sur les couches d'avant)
        else:
            return (self.weights - eta * weight_influence)[:, 0:self._input_size]

    @property
    def learning_batch_size(self):
        return self._learning_batch_size

    @learning_batch_size.setter
    def learning_batch_size(self, new_learning_batch_size):
        self.activation_levels = np.zeros((self._output_size, new_learning_batch_size))
        self.output = np.zeros((self._output_size, new_learning_batch_size))
        self.noise_input = np.zeros((self._noise_size, new_learning_batch_size))
        self._learning_batch_size = new_learning_batch_size


class ConvolutionalLayer(NeuronLayer):

    def __init__(self, activation_function, input_size=(1, 1), output_size=(1, 1),
                 learning_batch_size=1, filter_size=(1, 1), input_feature_maps=1,
                 output_feature_maps=1, convolution_mode='valid', step=1):
        super(ConvolutionalLayer, self).__init__(activation_function, 1, 1, learning_batch_size)
        self._filter_size = filter_size
        self._input_feature_maps = input_feature_maps
        self._output_feature_maps = output_feature_maps
        self._step = step  # Laisser à 1 pour l'instant
        self._convolution_mode = convolution_mode
        self._weights = np.random.randn(self._output_feature_maps, self._input_feature_maps,
                                        self._filter_size[0], self._filter_size[1])
        self._bias = np.zeros(self._output_feature_maps)
        self._input_size = input_size
        if self._convolution_mode == 'full':
            self._output_size = (self._input_size[0] + (self._filter_size[0]-1),
                                 self._input_size[1] + (self._filter_size[1]-1))
            self._reverse_convolution_mode = 'valid'
        # elif self._convolution_mode == 'same':
        #     self._output_size = self._input_size
        #     self._reverse_convolution_mode = 'same'
        elif self._convolution_mode == 'valid':
            self._output_size = (self._input_size[0] - (self._filter_size[0]-1),
                                 self._input_size[1] - (self._filter_size[1]-1))
            self._reverse_convolution_mode = 'full'
        else:
            raise Exception("Invalid convolution mode")
        self.input = np.zeros((self._learning_batch_size, self._input_feature_maps,
                               self._input_size[0], self._input_size[1]))
        self.activation_levels = np.zeros((self._learning_batch_size, self._output_feature_maps,
                                           self._output_size[0], self._output_size[1]))
        self.output = np.zeros((self._learning_batch_size, self._output_feature_maps,
                                self._output_size[0], self._output_size[1]))

    def compute(self, inputs):
        self.input = self.tensorize_inputs(inputs)
        conv = conv2d(self.input, self._weights, border_mode=self._convolution_mode)
        self.activation_levels = conv.eval() + self._bias[np.newaxis, :, np.newaxis, np.newaxis]
        self.output = self._activation_function.out(self.activation_levels)
        return self.output

    def calculate_weight_influence(self, out_influence):
        output_shape = (self._output_feature_maps, self._input_feature_maps,
                        self._filter_size[0], self._filter_size[1])
        weight_influence = conv2d(np.transpose(self.input, axes=(1, 0, 2, 3)),
                                  np.transpose(out_influence, axes=(1, 0, 2, 3)),
                                  border_mode=self._convolution_mode,
                                  filter_flip=False)
        return np.transpose(weight_influence.eval(), axes=(1, 0, 2, 3)) / self._learning_batch_size

    def calculate_bias_influence(self, out_influence):
        return np.mean(out_influence, axis=(0, 2, 3))

    def derivate_error(self, in_influence):
        deriv_vector = self._activation_function.derivate(self.activation_levels)
        return deriv_vector * self.tensorize_outputs(in_influence)

    def input_error(self, out_influence):
        output_shape = (self._learning_batch_size, self._input_feature_maps,
                        self._input_size[0], self._input_size[1])
        conv = conv2d(out_influence,
                      np.transpose(self._weights, axes=(1, 0, 2, 3)),
                      border_mode=self._reverse_convolution_mode,
                      filter_flip=False)
        return conv.eval()

    def tensorize_inputs(self, inputs):
        """
        Create a tensor for convolutional layers from a batch of flattened inputs

        This method should be called during each compute or backprop of the layer. Return a reshaped
        input with shape (learning_batch_size, input_feature_maps, input_size[0], input_size[1])

        :param inputs: A 4D tensor as a batch of 3D input tensors, or a matrix as a batch
        of flattened inputs
        :return: A 4D tensor as a batch 3D input tensors
        """
        ndim = inputs.ndim
        shape = inputs.shape
        if ndim == 4:
            return inputs
        elif ndim == 2:
            # check with self dimension (input_shape, input_channels), then reshape
            if self._input_size[0]*self._input_size[1]*self._input_feature_maps != shape[0]:
                raise Exception('Wrong dimensions : cannot reshape')
            inputs_reshaped = inputs.ravel('F').reshape((self._learning_batch_size,
                                                        self._input_feature_maps,
                                                        self._input_size[0],
                                                        self._input_size[1]))
            return inputs_reshaped
        else:
            raise Exception('Wrong inputs dimension, inputs should be a 4D tensor with '
                            'shape : (batch_size, inputs_channel, img_h, img_w), or a matrix of'
                            'flattened inputs')

    def tensorize_outputs(self, outputs):
        """
        Create a tensor for convolutional layers from a batch of flattened outputs

        This method should be called during each backprop of the layer. Return a reshaped
        output with shape (learning_batch_size, output_feature_maps, output_size[0], output_size[1])

        :param outputs: A 4D tensor as a batch of 3D output tensors, or a matrix as a batch
        of flattened outputs
        :return: A 4D tensor as a batch 3D output tensors
        """
        ndim = outputs.ndim
        shape = outputs.shape
        if ndim == 4:
            return outputs
        elif ndim == 2:
            outputs_reshaped = outputs.ravel('F').reshape((self._learning_batch_size,
                                                          self._output_feature_maps,
                                                          self._output_size[0],
                                                          self._output_size[1]))
            return outputs_reshaped
        else:
            raise Exception('Wrong inputs dimension, inputs should be a 4D tensor with '
                            'shape : (batch_size, inputs_channel, img_h, img_w), or a matrix of'
                            'flattened inputs')
