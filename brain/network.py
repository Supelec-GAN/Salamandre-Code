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
    def __init__(self, layers_neuron_count, layers_activation_function, error_function,
                 learning_batch_size=1, weights_list=()):
        self._layers_activation_function = layers_activation_function  # sauvegarde pour pouvoir
        # reinitialiser
        self.layers_neuron_count = layers_neuron_count
        self._layers_count = np.size(layers_neuron_count) - 1
        self.error = error_function
        self._learning_batch_size = learning_batch_size
        self.layers_list = np.array(
                            self._layers_count * [NeuronLayer(
                                                        self._layers_activation_function[0],
                                                        self.error
                                                        )]
                            )
        for i in range(0, self._layers_count):
            self.layers_list[i] = NeuronLayer(self._layers_activation_function[i],
                                              self.error,
                                              self.layers_neuron_count[i],
                                              self.layers_neuron_count[i + 1],
                                              self._learning_batch_size
                                              )
        self.output = np.zeros(self.layers_neuron_count[-1])

        if len(weights_list) != 0:  # si l'on a donné une liste de poids
            for i in range(0, self._layers_count):
                self.layers_list[i].weights = weights_list[i][0]
                self.layers_list[i].bias = weights_list[i][1]

    def reset(self):
        self.layers_list = np.array(
            self._layers_count * [NeuronLayer(
                self._layers_activation_function[0],
                self.error
            )]
        )
        for i in range(0, self._layers_count):
            self.layers_list[i] = NeuronLayer(self._layers_activation_function[i],
                                              self.error,
                                              self.layers_neuron_count[i],
                                              self.layers_neuron_count[i + 1],
                                              self._learning_batch_size
                                              )
        self.output = np.zeros(self.layers_neuron_count[-1])

    ##
    # @brief      On calcule la sortie du réseau
    #
    # @param      self    The object
    # @param      inputs  The inputs
    #
    # @return     La sortie de la dernière couche est la sortie finale
    #

    def compute(self, inputs):
        dim = np.shape(inputs)
        nb_dim = len(dim)
        if nb_dim == 1:     # Pour conserver le fonctionnement avec un vecteur simple en entrée
            inputs = np.reshape(inputs, (dim[0], 1))
        elif nb_dim == 2:
            inputs = np.reshape(inputs, (dim[1], dim[0]))
        else:
            raise Exception("Incorrect inputs dimensions")
        self.layers_list[0].compute(inputs)
        for i in range(1, self._layers_count):
            self.layers_list[i].compute(self.layers_list[i - 1].output)
        return self.layers_list[-1].output

    ##
    # @brief      Calcul d'erreur quadratique
    #
    # @param      x  la sortie à comparer
    # @param      reference  The reference
    #
    # @return     norme2 de la différence de vecteur
    #

    def backprop(self, eta, inputs, reference):
        inputs = np.reshape(inputs, (len(inputs), 1))
        n = self._layers_count
        # Si l'entrée et la sortie sont la même couche
        if n == 1:
            input_layer = inputs
        else:
            input_layer = self.layers_list[-2].output

        # On commence par la couche de sortie, avec initialisation de l'influence de la sortie
        out_influence = self.layers_list[n - 1].init_derivate_error(reference)
        self.layers_list[n - 1].backprop(out_influence, eta, input_layer)

        # On remonte la propagation jusqu'à la 2ème couche (si elle existe)
        for i in range(n - 2, 0, -1):
            input_layer = self.layers_list[i - 1].output

            out_influence = self.layers_list[i].derivate_error(
                out_influence,
                self.layers_list[i + 1].weights
            )
            self.layers_list[i].backprop(out_influence, eta, input_layer)

        # On s'occupe de la couche d'entrée (si différente de couche de sortie)
        if n > 1:
            input_layer = inputs

            out_influence = self.layers_list[0].derivate_error(
                out_influence,
                self.layers_list[1].weights
            )
            self.layers_list[0].backprop(out_influence, eta, input_layer)
        return out_influence

    ##
    # @brief      { function_description }
    #
    # @param      self       The object
    # @param      eta        The eta
    # @param      inputs     The inputs
    # @param      reference  The reference
    #
    # @return     { description_of_the_return_value }
    ##
    def no_update_backprop(self, eta, inputs, reference):
        inputs = np.reshape(inputs, (len(inputs), 1))
        n = self._layers_count
        # Si l'entrée et la sortie sont la même couche
        if n == 1:
            input_layer = inputs
        else:
            input_layer = self.layers_list[-2].output

        # On commence par la couche de sortie, avec initialisation de l'influence de la sortie
        out_influence = self.layers_list[n - 1].no_update_init_derivate_error(reference)
        next_weights = self.layers_list[n - 1].no_update_backprop(out_influence, eta, input_layer)

        # On remonte la propagation jusqu'à la 2ème couche (si elle existe)
        for i in range(n - 2, 0, -1):
            input_layer = self.layers_list[i - 1].output

            out_influence = self.layers_list[i].derivate_error(
                out_influence,
                next_weights
            )
            next_weights = self.layers_list[i].no_update_backprop(out_influence, eta, input_layer)

        # On s'occupe de la couche d'entrée (si différente de couche de sortie)
        if n > 1:
            input_layer = inputs

            out_influence = self.layers_list[0].derivate_error(
                out_influence,
                next_weights
            )
            self.layers_list[0].no_update_backprop(out_influence, eta, input_layer)
        return out_influence

    def save_state(self):
        """
        Permet de sauvegarder l'état du réseau, ainsi que ses paramètres

        Renvoie un tuple (paramètres, poids/biais). Les paramètres sont dans le même ordre que
        pour la création d'un Network. Les poids/biais sont une liste de tuple (poids,
        biais) correspondant au couche successives.
        """

        saved_activation_functions = []
        for f in self._layers_activation_function:
            saved_activation_functions.append(f.save_fun())
        saved_activation_functions = str(saved_activation_functions).replace("'", "")  # permet
        # d'avoir "[Sigmoid(mu), ...]", à la place de "['Sigmoid(mu)', ...]"
        params = [self.layers_neuron_count, saved_activation_functions, self.error.save_fun()]
        coefs = []
        for i in range(self._layers_count):
            layer_coefs = (self.layers_list[i].weights, self.layers_list[i].bias)
            coefs.append(layer_coefs)
        state = [params, coefs]
        return state
