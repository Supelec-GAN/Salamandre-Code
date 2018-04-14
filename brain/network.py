import numpy as np
from brain.neuronLayer import NeuronLayer, ConvolutionalLayer
from fonction import Norm2, NonSatHeuristic


class Network:
    """Classe permettant de créer un perceptron multicouche"""

    def __init__(self, layers_parameters, param_desc='', error_function=Norm2(), learning_batch_size=1,
                 error_gen=NonSatHeuristic(), nb_exp=0, weights_list=()):
        """
        Contruit un réseau de neurones avec des poids initialisés uniformément entre 0 et 1

        :param layers_parameters: Liste des paramètres de couches
        :param error_function: Fonction d'erreur du réseau
        :param learning_batch_size: Taille des batchs
        :param error_gen: Fonction d'erreur utilisée par le GAN pendant la rétropropagation sans
        mise à jour dans le discriminateur lors de l'appprentissage du générateur
        :param weights_list: Liste de poids/biais à renseigner si l'on veut restaurer un ancien
        réseau
        """

        self._layers_parameters = layers_parameters  # sauvegarde pour pouvoir réinitialiser
        self._layers_count = len(layers_parameters)
        self.param_desc = param_desc
        self._error = error_function
        self._error_gen = error_gen
        self.nb_exp = nb_exp
        self._learning_batch_size = learning_batch_size
        self.layers_list = np.array(self._layers_count * [NeuronLayer()])
        for i in range(0, self._layers_count):
            params = self._layers_parameters[i]
            if params['type'] == 'N':
                self.layers_list[i] = \
                    NeuronLayer(params['activation_function'],
                                input_size=params['input_size'],
                                output_size=params['output_size'],
                                noise_size=params['noise_size'],
                                nb_exp=self.nb_exp,
                                learning_batch_size=self._learning_batch_size
                                )
            elif params['type'] == 'C':
                self.layers_list[i] = \
                    ConvolutionalLayer(params['activation_function'],
                                       input_size=params['input_size'],
                                       output_size=params['output_size'],
                                       filter_size=params['filter_size'],
                                       input_feature_maps=params['input_feature_maps'],
                                       output_feature_maps=params['output_feature_maps'],
                                       convolution_mode=params['convolution_mode'],
                                       learning_batch_size=self._learning_batch_size
                                       )
            else:
                raise Exception('Wrong layer type')
        self.output = np.zeros(self.layers_list[-1].output_size)

        if len(weights_list) != 0:  # si l'on a donné une liste de poids
            for i in range(0, self._layers_count):
                self.layers_list[i].weights = weights_list[i][0]
                self.layers_list[i].bias = weights_list[i][1]

    def reset(self):
        """
        Réinitialise un réseau de neurones (poids uniformément répartis entre 0 et 1, biais nuls)

        :return: None
        """

        for i in range(0, self._layers_count):
            params = self._layers_parameters[i]
            if params['type'] == 'N':
                self.layers_list[i] = \
                    NeuronLayer(params['activation_function'],
                                input_size=params['input_size'],
                                output_size=params['output_size'],
                                noise_size=params['noise_size'],
                                nb_exp=self.nb_exp,
                                learning_batch_size=self._learning_batch_size
                                )
            elif params['type'] == 'C':
                self.layers_list[i] = \
                    ConvolutionalLayer(params['activation_function'],
                                       input_size=params['input_size'],
                                       output_size=params['output_size'],
                                       filter_size=params['filter_size'],
                                       input_feature_maps=params['input_feature_maps'],
                                       output_feature_maps=params['output_feature_maps'],
                                       convolution_mode=params['convolution_mode'],
                                       learning_batch_size=self._learning_batch_size
                                       )
            else:
                raise Exception('Wrong layer type')
        self.output = np.zeros(self.layers_list[-1].output_size)

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
    def backprop(self, reference, update=True, gen_backprop=False):
        """
        Rétropropagation selon la méthode de la descente du gradient

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
            self.layers_list[i].backprop(out_influence, update)
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

# A mettre à jour avec les nouveaux attributs !!
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
