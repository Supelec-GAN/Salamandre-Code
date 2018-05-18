
import numpy as np
from function.activationFunction import Function
from scipy.signal import convolve2d
from dataInterface import DataInterface


class NeuronLayer:

    def __init__(self, activation_function=Function(), input_size=1, output_size=1, noise_size=0,
                 learning_batch_size=1, param_desc='Parametres de descente', nb_exp=0):
        """
        Creates a fully connected neuron layer

        :param activation_function:
        :param input_size:
        :param output_size:
        :param noise_size:
        :param learning_batch_size:
        :param param_desc:
        :param nb_exp:
        """
        self._input_size = input_size
        self._output_size = output_size
        self._learning_batch_size = learning_batch_size
        self._noise_size = noise_size
        # self._weights = np.transpose(np.random.randn(input_size, output_size))
        self._weights = np.random.randn(output_size, input_size+noise_size)
        self._bias = np.zeros((output_size, 1))                                # Vecteur colonne
        # On peut laisser le biais comme un vecteur colonne, car en faire une matrice contenant
        # learning_batch_size fois la même colonne. Lorsque l'on aura besoin du biais dans les
        # calculs, il y aura mathématiquement parlant un problème de dimension (addition vecteur
        # + matrice), cependant numpy gère ça en additionnant le vecteur de biais à chacune des
        # colonnes de la matrice (broadcast)
        self.input = np.zeros((input_size, learning_batch_size))
        self._activation_function = activation_function
        self._activation_function.vectorize()
        self.activation_levels = np.zeros((output_size, learning_batch_size))  # Chaque colonne
        # correspond à une entrée du batch
        self.output = np.zeros((output_size, learning_batch_size))             # Chaque colonne
        # correspond à une entrée du batch

        self.update_weights_value = np.zeros((output_size, input_size + noise_size))
        self.update_bias_value = np.zeros((output_size, 1))

        self.noise_input = np.zeros((noise_size, learning_batch_size))

        # self.update_weights_value = np.zeros((output_size, input_size))

        self.weights_gradients_sum = np.zeros((output_size, input_size + noise_size))
        # self.weights_gradients_sum = np.zeros((output_size, input_size))
        self.bias_gradients_sum = np.zeros((output_size, 1))
        self.weights_moment = np.zeros((output_size, input_size + noise_size))
        # self.weights_moment = np.zeros((output_size, input_size))
        self.bias_moment = np.zeros((output_size, 1))
        self.weights_eta = np.zeros((output_size, input_size + noise_size))
        # self.weights_eta = np.zeros((output_size, input_size))            # need meilleur nom
        self.bias_eta = np.zeros((output_size, 1))                          # need meilleur nom

        data_interface = DataInterface()
        param_liste = data_interface.read_conf('config_algo_descente.ini', param_desc)  # Lecture
        # du fichier de config
        param_liste = data_interface.extract_param(param_liste, nb_exp)
        self.algo_utilise = param_liste['algo_utilise']
        self.eta = param_liste['eta']
        self.momentum = param_liste['momentum']
        self.epsilon = param_liste['epsilon']
        self.gamma = param_liste['gamma']
        self.moment = param_liste['moment']
        self.eta = param_liste['eta']
        self.gamma_1 = param_liste['gamma_1']
        self.gamma_2 = param_liste['gamma_2']
        self.instant = 0

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
        self.input = np.zeros((self._input_size, new_learning_batch_size))
        self.activation_levels = np.zeros((self._output_size, new_learning_batch_size))
        self.output = np.zeros((self._output_size, new_learning_batch_size))
        self.noise_input = np.zeros((self._noise_size, new_learning_batch_size))
        self._learning_batch_size = new_learning_batch_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    def compute(self, inputs):
        """
        Calcul de la sortie de la couche

        :param inputs: Nouvelles entrées
        :return: La sortie de la couche
        """
        self.input = self.flatten_inputs(inputs)
        if self._noise_size != 0:  # nécessaire car np.zeros((0,1)) est un objet chelou
            self.noise_input = np.random.randn(self._noise_size, self._learning_batch_size)
            self.input = np.concatenate([self.input, self.noise_input])
        self.activation_levels = np.dot(self._weights, self.input) - self._bias
        self.output = self._activation_function.out(self.activation_levels)
        return self.output

    def backprop(self, out_influence, update=True):
        """
        Rétropropagation au niveau d'une couche

        :param out_influence:
        :param update: Si Vrai, les poids de la couche sont mis à jour
        :return: Les nouveaux poids
        """
        weight_influence = self.calculate_weight_influence(out_influence)
        bias_influence = self.calculate_bias_influence(out_influence)
        if update:
            self.update_momentum(bias_influence, weight_influence)
            self.update_weights()
            self.update_bias()
            return self.weights[:, 0:self._input_size]  # On extrait les poids concernant les vrais
            # inputs (le bruit n'a pas besoin d'influer sur les couches d'avant)
        else:
            return (self.weights - self.eta * weight_influence)[:, 0:self._input_size]

    def update_momentum(self, bias_influence, weight_influence):
        """

        :param bias_influence:
        :param weight_influence:
        :return:
        """

        if self.algo_utilise == "Gradient":
            self.update_weights_value = self.momentum*self.update_weights_value - self.eta*weight_influence
            self.update_bias_value = self.momentum*self.update_bias_value + self.eta * bias_influence

        elif self.algo_utilise == "Adagrad":
            self.weights_gradients_sum = self.weights_gradients_sum + weight_influence**2
            partial = np.sqrt(np.add(self.weights_gradients_sum, self.epsilon))
            self.update_weights_value = self.momentum*self.update_weights_value - self.eta*np.divide(weight_influence, partial)

            self.bias_gradients_sum = self.bias_gradients_sum + bias_influence**2
            partial = np.sqrt(np.add(self.bias_gradients_sum, self.epsilon))
            self.update_bias_value = self.momentum*self.update_bias_value + self.eta*np.divide(bias_influence, partial)

        elif self.algo_utilise == "RMSProp":

            if self.moment == 2:
                self.weights_gradients_sum = self.gamma*self.weights_gradients_sum + (1 - self.gamma)*(weight_influence**2)
                partial = np.sqrt(np.add(self.weights_gradients_sum, self.epsilon))
                self.update_weights_value = self.momentum*self.update_weights_value - self.eta*np.divide(weight_influence, partial)

                self.bias_gradients_sum = self.gamma*self.bias_gradients_sum + (1-self.gamma)*(bias_influence**2)
                partial = np.sqrt(np.add(self.bias_gradients_sum, self.epsilon))
                self.update_bias_value = self.momentum*self.update_bias_value + self.eta*np.divide(bias_influence, partial)

            if self.moment == 1:
                self.weights_gradients_sum = self.gamma * self.weights_gradients_sum + (1 - self.gamma) * weight_influence ** 2
                self.weights_moment = self.gamma * self.weights_moment + (1- self.gamma) * weight_influence
                partial = np.sqrt(np.add((self.weights_gradients_sum - self.weights_moment**2), self.epsilon))
                self.update_weights_value = self.momentum*self.update_weights_value - self.eta*np.divide(weight_influence, partial)

                self.bias_gradients_sum = self.gamma * self.bias_gradients_sum + (1 - self.gamma) * bias_influence ** 2
                self.bias_moment = self.gamma * self.bias_moment + (1 - self.gamma) * bias_influence
                partial = np.sqrt(np.add((self.bias_gradients_sum - self.bias_moment**2), self.epsilon))
                self.update_bias_value = self.momentum*self.update_bias_value + self.eta*np.divide(bias_influence, partial)

        elif self.algo_utilise == "Adadelta":

            if self.moment == 2:
                self.weights_gradients_sum = self.gamma * self.weights_gradients_sum + (1 - self.gamma) * weight_influence ** 2
                self.weights_eta = self.gamma * self.weights_eta + (1 - self.gamma) * self.update_weights_value**2
                partial = np.sqrt(np.add(self.weights_eta, self.epsilon))*weight_influence
                partial2 = np.sqrt(np.add(self.weights_gradients_sum, self.epsilon))
                self.update_weights_value = self.momentum*self.update_weights_value - np.divide(partial, partial2)

                self.bias_gradients_sum = self.gamma * self.bias_gradients_sum + (1 - self.gamma) * bias_influence ** 2
                self.bias_eta = self.gamma * self.bias_eta + (1 - self.gamma) * self.update_bias_value ** 2
                partial = np.sqrt(np.add(self.bias_eta, self.epsilon))*bias_influence
                partial2 = np.sqrt(np.add(self.bias_gradients_sum, self.epsilon))
                self.update_bias_value = self.momentum*self.update_bias_value + np.divide(partial, partial2)

            if self.moment == 1:

                self.weights_gradients_sum = self.gamma * self.weights_gradients_sum + (1 - self.gamma) * weight_influence ** 2
                self.weights_eta = self.gamma * self.weights_eta + (1 - self.gamma) * self.update_weights_value ** 2
                self.weights_moment = self.gamma * self.weights_moment + (1 - self.gamma) * weight_influence
                partial = np.sqrt(np.add(self.bias_eta, self.epsilon))*bias_influence
                partial2 = np.sqrt(np.add((self.weights_gradients_sum - self.weights_moment**2), self.epsilon))
                self.update_weights_value = self.momentum*self.update_weights_value - np.divide(partial, partial2)

                self.bias_gradients_sum = self.gamma * self.bias_gradients_sum + (1 - self.gamma) * bias_influence ** 2
                self.bias_eta = self.gamma * self.bias_eta + (1 - self.gamma) * self.update_bias_value ** 2
                self.bias_moment = self.gamma * self.bias_moment + (1 - self.gamma) * bias_influence
                partial = np.sqrt(np.add(self.bias_eta, self.epsilon))*bias_influence
                partial2 = np.sqrt(np.add((self.bias_gradients_sum - self.bias_moment**2), self.epsilon))
                self.update_bias_value = self.momentum*self.update_bias_value + np.divide(partial, partial2)

        elif self.algo_utilise == "Adam":

            self.instant += 1

            self.weights_gradients_sum = self.gamma_2 * self.weights_gradients_sum + (1 - self.gamma_2) * weight_influence ** 2
            self.weights_moment = self.gamma_1 * self.weights_moment + (1 - self.gamma_1) * weight_influence
            partial =(1 - self.gamma_1**self.instant)
            partial2 = np.sqrt(np.add(np.divide(self.weights_gradients_sum, (1 - self.gamma_2**self.instant)), self.epsilon))
            self.update_weights_value = self.momentum * self.update_weights_value - self.eta*np.divide(np.divide(self.weights_moment, partial), partial2)

            self.bias_gradients_sum = self.gamma * self.bias_gradients_sum + (1 - self.gamma) * bias_influence ** 2
            self.bias_moment = self.gamma * self.bias_moment + (1 - self.gamma) * bias_influence
            partial =(1 - self.gamma_1**self.instant)
            partial2 = np.sqrt(np.add(np.divide(self.bias_gradients_sum, (1 - self.gamma_2**self.instant)), self.epsilon))
            self.update_bias_value = self.momentum * self.update_bias_value + self.eta*np.divide(np.divide(self.bias_moment, partial), partial2)

    def update_weights(self):
        """
        Updates weights according to update_weights_value that was calculated previously

        :return: None
        """
        self._weights = self._weights + self.update_weights_value

    def update_bias(self):
        """
        Updates bias according to update_bias_value that was calculated previously

        :return: None
        """
        self._bias = self._bias + self.update_bias_value

    def calculate_weight_influence(self, out_influence):
        """
        Calculates the weights influence

        :param out_influence: influence of output on the error
        :return: matrix of the same dimension than weights
        """
        return np.dot(out_influence, np.transpose(self.input)) / self._learning_batch_size

    def calculate_bias_influence(self, out_influence):
        """
        Calculates the bias influence (which is out_influence)

        :param out_influence: influence of output on the error
        :return: vector of the same dimension than bias
        """
        mean_out_influence = np.mean(out_influence, axis=1, keepdims=True)
        return mean_out_influence

    def derivate_error(self, in_influence):
        """
        Calculates the error derivation

        :param in_influence: influence of the input of the next layer
        :return: error used in the recursive formula
        """
        deriv_vector = self._activation_function.derivate(self.activation_levels)
        return deriv_vector * in_influence

    def input_error(self, out_influence, new_weights):
        """
        Propagates the error from the activation levels to the inputs

        :param out_influence: influence of the activation levels
        :param new_weights: the updated weights
        :return: influence of the input
        """
        in_influence = np.dot(np.transpose(new_weights), out_influence)
        return in_influence

    def flatten_inputs(self, inputs):
        """
        Creates a input matrix from a 4D tensor of a convolutional layer

        :param inputs: A 4D tensor or a matrix
        :return: A matrix as a batch of flattened input vectors
        """
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


class ConvolutionalLayer(NeuronLayer):

    def __init__(self, activation_function, input_size=(1, 1), output_size=(1, 1),
                 learning_batch_size=1, filter_size=(1, 1), input_feature_maps=1,
                 output_feature_maps=1, convolution_mode='valid', step=1):
        """
        Creates a convolutional layer for a neural network

        :param activation_function:
        :param input_size:
        :param output_size:
        :param learning_batch_size:
        :param filter_size:
        :param input_feature_maps:
        :param output_feature_maps:
        :param convolution_mode:
        :param step:
        """
        super(ConvolutionalLayer, self).__init__(activation_function, input_size=1, output_size=1,
                                                 learning_batch_size=learning_batch_size)
        self._filter_size = filter_size
        self._input_feature_maps = input_feature_maps
        self._output_feature_maps = output_feature_maps
        self._step = step  # Laisser à 1 pour l'instant
        self._convolution_mode = convolution_mode
        self._weights = np.random.randn(self._output_feature_maps, self._input_feature_maps,
                                        self._filter_size[0], self._filter_size[1])
        self._bias = np.zeros(self._output_feature_maps)
        self._input_size = input_size
        self._output_size = output_size
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

    @property
    def learning_batch_size(self):
        return self._learning_batch_size

    @learning_batch_size.setter
    def learning_batch_size(self, new_learning_batch_size):
        self.input = np.zeros((new_learning_batch_size, self._input_feature_maps,
                               self._input_size[0], self._input_size[0]))
        self.activation_levels = np.zeros((new_learning_batch_size, self._output_feature_maps,
                                           self._output_size[0], self._output_size[1]))
        self.output = np.zeros((new_learning_batch_size, self._output_feature_maps,
                                self._output_size[0], self._output_size[1]))
        self._learning_batch_size = new_learning_batch_size

    def compute(self, inputs):
        self.input = self.tensorize_inputs(inputs)
        self.activation_levels = self.conv2d() + self._bias[np.newaxis, :, np.newaxis, np.newaxis]
        self.output = self._activation_function.out(self.activation_levels)
        return self.output

    def backprop(self, out_influence, update=True):
        """
        Rétropropagation au niveau d'une couche

        :param out_influence:
        :param update: Si Vrai, les poids de la couche sont mis à jour
        :return: Les nouveaux poids
        """
        weight_influence = self.calculate_weight_influence(out_influence)
        bias_influence = self.calculate_bias_influence(out_influence)
        self.update_weights_value = weight_influence
        self.update_bias_value = bias_influence
        if update:
            # self.update_momentum(bias_influence, weight_influence)
            self.update_weights()
            self.update_bias()
            return self.weights
        else:
            return self.weights - self.eta * weight_influence

    def calculate_weight_influence(self, out_influence):
        return self.weights_conv2d(out_influence) / self._learning_batch_size

    def calculate_bias_influence(self, out_influence):
        return np.mean(out_influence, axis=(0, 2, 3))

    def derivate_error(self, in_influence):
        deriv_vector = self._activation_function.derivate(self.activation_levels)
        return deriv_vector * self.tensorize_outputs(in_influence)

    def input_error(self, out_influence, new_weights):
        return self.reverse_conv2d(out_influence, new_weights)

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
        # shape = outputs.shape
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
                            'shape : (batch_size, outputs_channel, img_h, img_w), or a matrix of'
                            'flattened inputs')

    def conv2d(self):
        out = np.zeros_like(self.output)
        for b in range(self._learning_batch_size):
            for o in range(self._output_feature_maps):
                for i in range(self._input_feature_maps):
                    out[b][o] += convolve2d(self.input[b][i],
                                            self._weights[o][i],
                                            mode=self._convolution_mode)
        return out

    def reverse_conv2d(self, out_influence, new_weights):
        in_influence = np.zeros_like(self.input)
        for b in range(self._learning_batch_size):
            for i in range(self._input_feature_maps):
                for o in range(self._output_feature_maps):
                    in_influence[b][i] += convolve2d(out_influence[b][o],
                                                     np.rot90(new_weights[o][i], k=2),
                                                     mode=self._reverse_convolution_mode)
        return in_influence

    def weights_conv2d(self, out_influence):
        weight_influence = np.zeros_like(self._weights)
        for o in range(self._output_feature_maps):
            for i in range(self._input_feature_maps):
                for b in range(self._learning_batch_size):
                    weight_influence[o][i] += convolve2d(self.input[b][i],
                                                         np.rot90(out_influence[b][o], k=2),
                                                         mode=self._convolution_mode)
        return weight_influence


class ClippedNeuronLayer(NeuronLayer):

    def __init__(self, activation_function=Function(), input_size=1, output_size=1, noise_size=0,
                 learning_batch_size=1, param_desc='Parametres de descente', nb_exp=0, clipping=0):

        super(ClippedNeuronLayer, self).__init__(activation_function, input_size=input_size, output_size=output_size,
                                                 learning_batch_size=learning_batch_size,
                                                 param_desc=param_desc, nb_exp=nb_exp)

        self._clipping = clipping
        """
        Creates a Fully Connected layer with clipped weights for a neural network

        :param activation_function:
        :param input_size:
        :param output_size:
        :param learning_batch_size:
        :param clipping
        """

        def backprop(self, out_influence, update=True):
            """
            Rétropropagation au niveau d'une couche

            :param out_influence:
            :param update: Si Vrai, les poids de la couche sont mis à jour
            :return: Les nouveaux poids
            """
            weight_influence = self.calculate_weight_influence(out_influence)
            bias_influence = self.calculate_bias_influence(out_influence)
            if update:
                self.update_momentum(bias_influence, weight_influence)
                self.update_weights()
                self.update_bias()
                return self.weights[:, 0:self._input_size]  # On extrait les poids concernant les vrais
                # inputs (le bruit n'a pas besoin d'influer sur les couches d'avant)
            else:
                return (self.weights + self.eta * weight_influence)[:, 0:self._input_size]

                # on fait + self.eta, avec l'hypothèse que le clipping ne sert que pour WGAN pour le moment !!!

    def update_weights(self):
        """
        Updates weights according to update_weights_value that was calculated previously

        :return: None
        """
        self._weights = self._weights + self.update_weights_value
        self.weights_clipping()

    def update_bias(self):
        """
        Updates bias according to update_bias_value that was calculated previously

        :return: None
        """
        self._bias = self._bias + self.update_bias_value
        self.bias_clipping()

    def weights_clipping(self):
        """
        Clip the weight in [-clipping, +clipping] with linear transformations.

        :return: none
        """
        max_weigth = np.amax(np.abs(self._weights))
        self._weights = self._clipping*self._weights/max_weigth

    def bias_clipping(self):
        """
        Clip the basin [-clipping, +clipping] with linear transformations.
        
        :return: none
        """
        max_bias = np.amax(np.abs(self._bias))
        self._bias = self._clipping*self._bias/max_bias