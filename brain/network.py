from brain.neuronLayer import NeuronLayer, ConvolutionalLayer, ClippedNeuronLayer
from function.activationFunction import *
from function.errorFunction import *


class Network:

    def __init__(self, layers_parameters, error_function=Norm2(), learning_batch_size=1,
                 error_gen=NonSatHeuristic(), param_desc='Parametres de descente', nb_exp=0):
        """
        Contruit un réseau de neurones multicouches avec des poids initialisés uniformément entre
        0 et 1
s
        :param layers_parameters: Liste des paramètres de couches
        :param error_function: Fonction d'erreur du réseau
        :param learning_batch_size: Taille des batchs
        :param error_gen: Fonction d'erreur utilisée par le GAN pendant la rétropropagation sans
        mise à jour dans le discriminateur lors de l'appprentissage du générateur
        """

        self._layers_parameters = layers_parameters  # sauvegarde pour pouvoir réinitialiser
        self._layers_count = len(layers_parameters)
        self._error = error_function
        self._error_gen = error_gen
        self._param_desc = param_desc
        self.nb_exp = nb_exp
        self._learning_batch_size = learning_batch_size
        self.layers_list = np.array(self._layers_count * [NeuronLayer()])
        for i in range(0, self._layers_count):
            params = self._layers_parameters[i]
            if params['type'] == 'FC':
                self.layers_list[i] = \
                    NeuronLayer(activation_function=eval(params['activation_function']),
                                input_size=params['input_size'],
                                output_size=params['output_size'],
                                noise_size=params['noise_size'],
                                param_desc=self._param_desc,
                                nb_exp=self.nb_exp,
                                learning_batch_size=self._learning_batch_size
                                )
            elif params['type'] == 'Conv':
                self.layers_list[i] = \
                    ConvolutionalLayer(activation_function=eval(params['activation_function']),
                                       input_size=params['input_size'],
                                       output_size=params['output_size'],
                                       filter_size=params['filter_size'],
                                       input_feature_maps=params['input_feature_maps'],
                                       output_feature_maps=params['output_feature_maps'],
                                       convolution_mode=params['convolution_mode'],
                                       learning_batch_size=self._learning_batch_size
                                       )
            elif params['type'] == 'Clipped':
                self.layers_list[i] = \
                    ClippedNeuronLayer(activation_function=eval(params['activation_function']),
                                       input_size=params['input_size'],
                                       output_size=params['output_size'],
                                       noise_size=params['noise_size'],
                                       param_desc=self._param_desc,
                                       nb_exp=self.nb_exp,
                                       learning_batch_size=self._learning_batch_size,
                                       clipping=params['clipping']
                                       )
            else:
                raise Exception('Wrong layer type')
            try:
                coefs = params['coefs']
                self.layers_list[i].restore_coefs(coefs)
            except KeyError:
                pass
        self.input_size = np.prod(self.layers_list[0].input_size)
        self.input = np.zeros((self.input_size, self._learning_batch_size))
        self.output_size = np.prod(self.layers_list[-1].output_size)
        self.output = np.zeros((self.output_size, self._learning_batch_size))

    def reset(self):
        """
        Réinitialise un réseau de neurones (poids uniformément répartis entre 0 et 1, biais nuls)

        :return: None
        """

        for i in range(0, self._layers_count):
            params = self._layers_parameters[i]
            if params['type'] == 'FC':
                self.layers_list[i] = \
                    NeuronLayer(activation_function=eval(params['activation_function']),
                                input_size=params['input_size'],
                                output_size=params['output_size'],
                                noise_size=params['noise_size'],
                                param_desc=self._param_desc,
                                nb_exp=self.nb_exp,
                                learning_batch_size=self._learning_batch_size
                                )
            elif params['type'] == 'Conv':
                self.layers_list[i] = \
                    ConvolutionalLayer(activation_function=eval(params['activation_function']),
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
    def backprop(self, reference, update=True, gen_backprop=False, calculate_error=True):
        """
        Rétropropagation selon la méthode de la descente du gradient

        :param reference: Sortie idéale
        :param update: Si vrai, on met à jour les poids/biais, sinon on ne renvoie que l'influence
        de l'erreur sur l'entrée
        :param gen_backprop: Dans le cas du GAN, indique d'utiliser _error_gen à la place de _error
        :return: Influence de l'erreur sur l'entrée
        """
        # On initialise avec une valeur particulière pour la couche de sortie
        if calculate_error:
            if gen_backprop:
                in_influence = self._error_gen.derivate(reference)  # reference = self.output ici
            else:
                in_influence = self._error.derivate(reference, self.output)
        else:
            in_influence = reference
        n = self._layers_count
        for i in range(n - 1, -1, -1):
            out_influence = self.layers_list[i].derivate_error(in_influence)
            new_weights = self.layers_list[i].backprop(out_influence, update)
            in_influence = self.layers_list[i].input_error(out_influence, new_weights)
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

        :return: Un triplet (layer_params, error_fun, gen_error_fun) avec layer_params contenznt
        les coéfficients de chaque couche..
        """

        for i in range(self._layers_count):
            coefs = self.layers_list[i].save_coefs()
            self._layers_parameters[i]['coefs'] = coefs
        state = [self._layers_parameters, self._error.save_fun(), self._error_gen.save_fun()]
        return state
