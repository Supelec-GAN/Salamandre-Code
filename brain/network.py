import numpy as np
from brain.neuronLayer import NeuronLayer


class Network:
    """Classe permettant de créer un perceptron multicouche"""

    ##
    # @brief      Constructs the object.
    #
    # @param      self                        The object
    # @param      layers_neuron_count         Nombre de Neurones par couches,
    #                                         en incluant le nombres d'entrées en position 0
    # @param      layers_activation_function  The layers activation function
    #
    def __init__(self, layers_neuron_count, layers_activation_function, error_function):
        self._layers_activation_function = layers_activation_function  # sauvegarde pour pouvoir reinitialiser
        self._layers_neuron_count = layers_neuron_count
        self._layers_count = np.size(layers_neuron_count) - 1
        self.error = error_function
        self._layers_list = np.array(
                            self._layers_count * [NeuronLayer(
                                                        layers_activation_function[0],
                                                        error_function
                                                        )]
                            )
        for i in range(0, self._layers_count):
            self._layers_list[i] = NeuronLayer(layers_activation_function[i],
                                               self.error,
                                               layers_neuron_count[i],
                                               layers_neuron_count[i + 1]
                                               )
        self.output = np.zeros(layers_neuron_count[-1])

    def reset(self):
        self._layers_list = np.array(
                            self._layers_count * [NeuronLayer(
                                                        self._layers_activation_function[0],
                                                        self.error
                                                        )]
                            )
        for i in range(0, self._layers_count):
            self._layers_list[i] = NeuronLayer(self._layers_activation_function[i],
                                               self.error,
                                               self._layers_neuron_count[i],
                                               self._layers_neuron_count[i + 1]
                                               )
        self.output = np.zeros(self._layers_neuron_count[-1])
    ##
    # @brief      On calcule la sortie du réseau
    #
    # @param      self    The object
    # @param      inputs  The inputs
    #
    # @return     La sortie de la dernière couche est la sortie finale
    #

    def compute(self, inputs):
        inputs = np.reshape(inputs, (len(inputs), 1))
        self._layers_list[0].compute(inputs)
        for i in range(1, self._layers_count):
            self._layers_list[i].compute(self._layers_list[i - 1].output)
        return self._layers_list[-1].output

    # @staticmethod
    # ##
    # # @brief      Calcul d'erreur quadratique
    # #
    # # @param      x  la sortie à comparer
    # # @param      reference  The reference
    # #
    # # @return     norme2 de la différence de vecteur
    # #
    # def error(x, reference):
    #     return np.linalg.norm(x - reference)

    def backprop(self, eta, inputs, reference):
        inputs = np.reshape(inputs, (len(inputs), 1))
        n = self._layers_count
        # Si l'entrée et la sortie sont la même couche
        if n == 1:
            input_layer = inputs
        else:
            input_layer = self._layers_list[-2].output

        # On commence par la couche de sortie, avec initialisation de l'influence de la sortie
        out_influence = self._layers_list[n-1].init_derivate_error(reference)
        self._layers_list[n-1].backprop(out_influence, eta, input_layer)

        # On remonte la propagation jusqu'à la 2ème couche (si elle existe)
        for i in range(n-2, 0, -1):
            input_layer = self._layers_list[i - 1].output

            out_influence = self._layers_list[i].derivate_error(
                                                out_influence,
                                                self._layers_list[i+1].weights
                                                )
            self._layers_list[i].backprop(out_influence, eta, input_layer)

        # On s'occupe de la couche d'entrée (si différente de couche de sortie)
        if n > 1:
            input_layer = inputs

            out_influence = self._layers_list[0].derivate_error(
                                                out_influence,
                                                self._layers_list[1].weights
                                                )
            self._layers_list[0].backprop(out_influence, eta, input_layer)
