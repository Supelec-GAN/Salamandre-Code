import numpy as np
from brain.neuronLayer import NeuronLayer, NoisyLayer
from fonction import Norm2, NonSatHeuristic


class Network:
    """Classe permettant de créer un perceptron multicouche"""

    def __init__(self, layers_neuron_count, layers_activation_function, layers_type,
                 error_function=Norm2(), learning_batch_size=1, error_gen=NonSatHeuristic(),
                 weights_list=()):
        """
        Contruit un réseau de neurones avec des poids initialisés uniformément entre 0 et 1

        :param layers_neuron_count: Liste avec le nombre de neurones par couches, en incluant le
        nombre d'entrées en position 0
        :param layers_activation_function: Liste des fonctions d'activation de chaque couche
        :param layers_type: Liste des types de couches ('N': NeuronLayer, 'C': ConvolutionalLayer,
        'B': NoisyLayer)
        :param error_function: Fonction d'erreur du réseau
        :param learning_batch_size: Taille des batchs
        :param error_gen: Fonction d'erreur utilisée par le GAN pendant la rétropropagation sans
        mise à jour dans le discriminateur lors de l'appprentissage du générateur
        :param weights_list: Liste de poids/biais à renseigner si l'on veut restaurer un ancien
        réseau
        """

        self._layers_activation_function = layers_activation_function  # sauvegarde pour pouvoir
        # réinitialiser
        self.layers_neuron_count = layers_neuron_count
        self._layers_type = layers_type
        self._layers_count = np.size(layers_neuron_count) - 1
        self._error = error_function
        self._error_gen = error_gen
        self._learning_batch_size = learning_batch_size
        self.layers_list = np.array(self._layers_count * [NeuronLayer()])
        for i in range(0, self._layers_count):
            self.layers_list[i] = NeuronLayer(self._layers_activation_function[i],
                                              self.layers_neuron_count[i],
                                              self.layers_neuron_count[i + 1],
                                              self._learning_batch_size
                                              )
        self.output = np.zeros(layers_neuron_count[-1])

        if len(weights_list) != 0:  # si l'on a donné une liste de poids
            for i in range(0, self._layers_count):
                self.layers_list[i].weights = weights_list[i][0]
                self.layers_list[i].bias = weights_list[i][1]

    def reset(self):
        """
        Réinitialise un réseau de neurones (poids uniformément répartis entre 0 et 1, biais nuls)

        :return: None
        """

        self.layers_list = np.array(self._layers_count * [NeuronLayer()])
        for i in range(0, self._layers_count):
            self.layers_list[i] = NeuronLayer(self._layers_activation_function[i],
                                              self.layers_neuron_count[i],
                                              self.layers_neuron_count[i + 1],
                                              self._learning_batch_size
                                              )
        self.output = np.zeros(self.layers_neuron_count[-1])

    def compute(self, inputs):
        """
        Calcule la sortie du réseau pour un batch d'entrées

        :param inputs: Batch d'entrées, sous le format (input_size, batch_size) (sera reshape ou
        transposé si nécessaire
        :return: Sortie du réseau, c'est-à-dire la sortie de la dernière couche
        """

        ndim = inputs.ndim
        shape = inputs.shape
        if ndim == 1:  # and self._learning_batch_size == 1:  # Pour conserver le fonctionnement
            # avec un vecteur simple en entrée
            inputs = np.reshape(inputs, (shape[0], 1))
        elif ndim == 2 and self._learning_batch_size == shape[0]:
            inputs = np.transpose(inputs)
        elif ndim == 2 and self._learning_batch_size == shape[1]:
            inputs = inputs
        else:
            raise Exception("Incorrect inputs dimensions")

        self.layers_list[0].compute(inputs)
        for i in range(1, self._layers_count):
            self.layers_list[i].compute(self.layers_list[i - 1].output)
        self.output = self.layers_list[-1].output
        return self.output

    # On considère ici toujours des réseaux avec plusieurs couches !
    # !! Rajouter la non_update_backprop
    def backprop(self, eta, reference, update=True, gen_backprop=False):
        """
        Rétropropagation selon la méthode de la descente du gradient

        :param eta: Coefficient d'apprentissage
        :param reference: Sortie idéale
        :param update: Si vrai, on met à jour les poids/biais, sinon on ne renvoie que l'influence
        de l'erreur sur l'entrée
        :param gen_backprop: Dans le cas du GAN, indique d'utiliser _error_gen à la place de _error
        :return: Influence de l'erreur sur l'entrée
        """
        # On initialise avec une valeur particulière pour la couche de sortie
        if gen_backprop:
            in_influence = self._error_gen.derivate(reference, self.output)
        else:
            in_influence = self._error.derivate(reference, self.output)

        n = self._layers_count
        for i in range(n - 1, -1, -1):
            out_influence = self.layers_list[i].derivate_error(in_influence)
            self.layers_list[i].backprop(out_influence, eta, update)
            in_influence = self.layers_list[i].input_error(out_influence)

        return in_influence

    @property
    def learning_batch_size(self):
        return self._learning_batch_size

    @learning_batch_size.setter
    def learning_batch_size(self, new_learning_batch_size):
        for layer in self.layers_list:
            layer.learning_batch_size = new_learning_batch_size
        self._learning_batch_size = new_learning_batch_size

    def save_state(self):
        """
        Permet de sauvegarder l'état du réseau, ainsi que ses paramètres

        :return: A tuple (paramètres, poids/biais). Les paramètres sont dans le même ordre que
        pour la création d'un Network. Les poids/biais sont une liste de tuple (poids,
        biais) correspondant au couche successives.
        """

        saved_activation_functions = []
        for f in self._layers_activation_function:
            saved_activation_functions.append(f.save_fun())
        saved_activation_functions = str(saved_activation_functions).replace("'", "")  # permet
        # d'avoir "[Sigmoid(mu), ...]", à la place de "['Sigmoid(mu)', ...]"
        params = [self.layers_neuron_count, saved_activation_functions, self._error.save_fun()]
        coefs = []
        for i in range(self._layers_count):
            layer_coefs = (self.layers_list[i].weights, self.layers_list[i].bias)
            coefs.append(layer_coefs)
        state = [params, coefs]
        return state


##
# @brief      Class for noisy generator network. It has NoisyLayer instead of NeuronLayer
##
class NoisyNetwork(Network):
    def __init__(self, layers_neuron_count, layers_activation_function, error_function,
                 noise_layers_size, learning_batch_size=1, weights_list=()):
        super(NoisyNetwork, self).__init__(layers_neuron_count, layers_activation_function, _,
                                           error_function, learning_batch_size, weights_list)
        self.noise_layers_size = noise_layers_size
        self.layers_list = np.array(
            self._layers_count * [NoisyLayer(self._layers_activation_function[0],
                                             self.layers_neuron_count[0],
                                             self.layers_neuron_count[1],
                                             self._learning_batch_size,
                                             self.noise_layers_size[0])]
        )
        for i in range(0, self._layers_count):
            self.layers_list[i] = NoisyLayer(self._layers_activation_function[i],
                                             self.layers_neuron_count[i],
                                             self.layers_neuron_count[i + 1],
                                             self._learning_batch_size,
                                             self.noise_layers_size[i]
                                             )

        self.output = np.zeros(layers_neuron_count[-1])

        if len(weights_list) != 0:  # si l'on a donné une liste de poids
            for i in range(0, self._layers_count):
                self.layers_list[i].weights = weights_list[i][0]
                self.layers_list[i].bias = weights_list[i][1]
