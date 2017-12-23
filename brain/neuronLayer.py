import numpy as np
from fonction import Sigmoid, MnistTest, Norm2


class NeuronLayer:
    """Classe permettant de créer une couche de neurones"""

    def __init__(self, activation_function, error_function, input_size=1, output_size=1):
        # Matrice de dimension q*p avec le nombre de sortie et p le nombre d'entrée
        self._input_size = input_size
        self._output_size = output_size
        self.weights = np.transpose(np.random.randn(input_size, output_size))
        self._bias = np.zeros((output_size, 1))            # Vecteur colonne
        self._activation_function = activation_function
        # self.error = error_function
        self.activation_levels = np.zeros((output_size, 1))  # Vecteur colonne
        self.output = np.zeros((output_size, 1))             # Vecteur colonne
        self.error = error_function

    @property
    def weights(self):
        """Get the current weights."""
        return self.weights

    @weights.setter
    def weights(self, newweights):
        self.weights = newweights

    @weights.deleter
    def weights(self):
        del self.weights

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

    ##
    # @brief      Calcul des sorties de la couche
    #
    # @param      inputs  Inputs

    def compute(self, inputs):
        self.activation_levels = np.dot(self.weights, inputs) - self._bias
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
    def backprop(self, out_influence, eta, input_layer):
        weight_influence = self.calculate_weight_influence(
            input_layer, out_influence)
        self.updateweights(eta, weight_influence)

        bias_influence = self.calculate_bias_influence(out_influence)
        self.update_bias(eta, bias_influence)

    def updateweights(self, eta, weight_influence):
        self.weights = self.weights - eta * weight_influence

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
    def calculate_weight_influence(self, input_layer, out_influence):
        return np.dot(out_influence, np.transpose(input_layer))

    ##
    # @brief      Calculates the bias influence (which is out_influence)
    def calculate_bias_influence(self, out_influence):
        return out_influence

    ##
    # @brief      Calculates the error derivation
    #
    # @param      out_influence  influence of output on the error
    #
    # @return     the error used in the recursive formula
    #
    def derivate_error(self, out_influence, nextweights):
        deriv_vector = self._activation_function.derivate(self.activation_levels)
        return deriv_vector * np.dot(np.transpose(nextweights), out_influence)

    ##
    # @brief      Initiate the error derivation
    #
    # @param      reference  the expected output for the last computation
    #
    # @return     {an derivative based by default on quadratic error}
    #
    def init_derivate_error(self, reference):
        deriv_vector = self._activation_function.derivate(self.activation_levels)
        return deriv_vector * self.error.derivate(reference, self.output)
