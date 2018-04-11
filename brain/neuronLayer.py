import numpy as np
from fonction import Sigmoid, MnistTest, Norm2, NonSatHeuristic
from math import sqrt
from dataInterface import DataInterface


class NeuronLayer:
    """Classe permettant de créer une couche de neurones"""

    def __init__(self, activation_function, error_function, param_desc, 
                 input_size=1, output_size=1, noise_size=0, learning_batch_size=1, nb_exp=0):
        # Matrice de dimension q*p avec le nombre de sortie et p le nombre d'entrée
        self._input_size = input_size
        self._output_size = output_size
        self._learning_batch_size = learning_batch_size
        # self._weights = np.transpose(np.random.randn(input_size, output_size))
        self._noise_size = noise_size

        self.weights = np.transpose(np.random.randn(input_size+noise_size, output_size))

        self._bias = np.zeros((output_size, 1))                                # Vecteur colonne
        # On peut laisser le biais comme un vecteur colonne, car en faire une matrice contenant
        # learning_batch_size fois la même colonne. Lorsque l'on aura besoin du biais dans les
        # calculs, il y aura mathématiquement parlant un problème de dimension (addition vecteur
        # + matrice), cependant numpy gère ça en additionnant le vecteur de biais à chacune des
        # colonnes de la matrice
        self._activation_function = activation_function
        self.activation_levels = np.zeros((output_size, learning_batch_size))  # Chaque colonne
        # correspond à une entrée du batch
        self.output = np.zeros((output_size, learning_batch_size))             # Chaque colonne
        # correspond à une entrée du batch
        self.error = error_function

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
        # self.weights_eta = np.zeros((output_size, input_size))          #need meilleur nom
        self.bias_eta = np.zeros((output_size, 1))                      #need meilleur nom

        data_interface = DataInterface()
        param_liste = data_interface.read_conf('config_algo_descente.ini', param_desc)  # Lecture du fichier de config
        param_liste = data_interface.extract_param(param_liste, nb_exp)
        self.algo_utilise = param_liste['algo_utilise']
        self.eta = param_liste['eta']
        self.momentum = param_liste['momentum']
        self.epsilon = param_liste['epsilon']
        self.gamma = param_liste['gamma']
        self.moment = param_liste['moment']
        self.alpha = param_liste['alpha']
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
        self.activation_levels = np.zeros((self._output_size, new_learning_batch_size))
        self.output = np.zeros((self._output_size, new_learning_batch_size))
        self.noise_input = np.zeros((self._noise_size, new_learning_batch_size))
        self._learning_batch_size = new_learning_batch_size

    def compute(self, inputs):
        if self._noise_size != 0:  # nécessaire car np.zeros( (0,1)) est un objet chelou
            self.noise_input = np.random.randn(self._noise_size, self._learning_batch_size)
            inputs = np.concatenate([inputs, self.noise_input])
        self.activation_levels = np.dot(self.weights, inputs) - self._bias
        self.output = self._activation_function.out(self.activation_levels)
        return self.output
    ##
    # @brief      Calcul des sorties de la couche
    #
    # @param      inputs  Inputs

    def backprop(self, out_influence, input_layer, update=True):
        if self._noise_size != 1:
            input_layer = np.concatenate([input_layer, self.noise_input])
        weight_influence = self.calculate_weight_influence(
            input_layer, out_influence)
        bias_influence = self.calculate_bias_influence(out_influence)
        if update:
            self.update_momentum(bias_influence, weight_influence)
            self.update_weights(weight_influence)
            self.update_bias(bias_influence)
            return self.weights[:, 0:self._input_size]  # On extrait les poids concernant les vrais inputs (le bruit n'a pas besoin d'influer sur les couches d'avant)
        else:
            return (self.weights - self.eta * weight_influence)[:, 0:self._input_size]


    def update_momentum(self, bias_influence, weight_influence):

        if self.algo_utilise == "Gradient":
            self.update_weights_value = self.momentum*self.update_weights_value - self.eta*weight_influence
            self.update_bias_value = self.momentum*self.update_bias_value - self.eta * bias_influence

        elif self.algo_utilise == "Adagrad":
            self.weights_gradients_sum = self.weights_gradients_sum + weight_influence**2
            partial = np.sqrt(np.add(self.weights_gradients_sum, self.epsilon))
            self.update_weights_value = self.momentum*self.update_weights_value - self.eta*np.divide(weight_influence, partial)

            self.bias_gradients_sum = self.bias_gradients_sum + bias_influence**2
            partial = np.sqrt(np.add(self.bias_gradients_sum, self.epsilon))
            self.update_bias_value = self.momentum*self.update_bias_value - self.eta*np.divide(weight_influence, partial)

        elif self.algo_utilise == "RMSProp":

            if self.moment == 2:
                self.weights_gradients_sum = self.gamma*self.weights_gradients_sum + (1 - self.gamma)*(weight_influence**2)
                partial = np.sqrt(np.add(self.weights_gradients_sum, self.epsilon))
                self.update_weights_value = self.momentum*self.update_weights_value - self.eta*np.divide(weight_influence, partial)

                # print(np.amax(partial))
                self.bias_gradients_sum = self.gamma*self.bias_gradients_sum + (1-self.gamma)*(bias_influence**2)
                partial = np.sqrt(np.add(self.bias_gradients_sum, self.epsilon))
                self.update_bias_value = self.momentum*self.update_bias_value - self.eta*np.divide(bias_influence, partial)

            if self.moment == 1:
                self.weights_gradients_sum = self.gamma * self.weights_gradients_sum + (1 - self.gamma) * weight_influence ** 2
                self.weights_moment = self.gamma * self.weights_moment + (1- self.gamma) * weight_influence
                partial = np.sqrt(np.add((self.weights_gradients_sum - self.weights_moment**2), self.epsilon))
                self.update_weights_value = self.momentum*self.update_weights_value - self.eta*np.divide(weight_influence, partial)
                
                self.bias_gradients_sum = self.gamma * self.bias_gradients_sum + (1 - self.gamma) * bias_influence ** 2
                self.bias_moment = self.gamma * self.bias_moment + (1 - self.gamma) * bias_influence
                partial = np.sqrt(np.add((self.bias_gradients_sum - self.bias_moment**2), self.epsilon))
                self.update_bias_value = self.momentum*self.update_bias_value - self.eta*np.divide(bias_influence, partial)

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
                self.update_bias_value = self.momentum*self.update_bias_value - np.divide(partial, partial2)

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
                self.update_bias_value = self.momentum*self.update_bias_value - np.divide(partial, partial2)

        elif self.algo_utilise == "Adam":

            self.instant += 1
            
            self.weights_gradients_sum = self.gamma * self.weights_gradients_sum + (1 - self.gamma) * weight_influence ** 2
            self.weights_moment = self.gamma * self.weights_moment + (1 - self.gamma) * weight_influence
            partial =(1 - self.gamma_1**self.instant)
            partial2 = np.sqrt(np.add(np.divide(self.weights_gradients_sum, (1 - self.gamma_2**self.instant)), self.epsilon))
            self.update_weights_value = self.momentum * self.update_weights_value - self.alpha*np.divide(np.divide(self.weights_moment, partial), partial2)
            
            self.bias_gradients_sum = self.gamma * self.bias_gradients_sum + (1 - self.gamma) * bias_influence ** 2
            self.bias_moment = self.gamma * self.bias_moment + (1 - self.gamma) * bias_influence
            partial =(1 - self.gamma_1**self.instant)
            partial2 = np.sqrt(np.add(np.divide(self.bias_gradients_sum, (1 - self.gamma_2**self.instant)), self.epsilon))
            self.update_bias_value = self.momentum * self.update_bias_value - self.alpha*np.divide(np.divide(self.bias_moment, partial), partial2)

    def update_weights(self, weight_influence):
        # self.update_weights_value = momentum*self.update_weights_value - eta * weight_influence
        self.weights = self.weights + self.update_weights_value

    def update_bias(self, bias_influence):
        # self.update_bias_value = momentum * self.update_bias_value + eta * bias_influence
        self._bias = self._bias + self.update_bias_value

    ##
    # @brief      Calculates the weight influence.
    #
    # @param      input_layer    input of the last compute
    # @param      out_influence  influence of output on the error
    #
    # @return     vecteur of same dimension than weights.
    #
    def calculate_weight_influence(self, input_layer, out_influence):
        return np.dot(out_influence, np.transpose(input_layer)) / self._learning_batch_size

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
    def derivate_error(self, out_influence, next_weights):
        deriv_vector = self._activation_function.derivate(self.activation_levels)
        return deriv_vector * np.dot(np.transpose(next_weights), out_influence)


##
# @brief      Class for output layer (different derivate error).
##
class OutputLayer(NeuronLayer):
    def __init__(self, activation_function, error_function, param_desc, input_size=1, output_size=1, learning_batch_size=1, nb_exp=0):
        super(OutputLayer, self).__init__(activation_function, error_function, param_desc, input_size, output_size, 0, learning_batch_size, nb_exp)

        data_interface = DataInterface()
        param_liste = data_interface.read_conf('config_algo_descente.ini', param_desc)  # Lecture du fichier de confi
        param_liste = data_interface.extract_param(param_liste, nb_exp)
        self.error_gen = param_liste['error_function_gen']

    def derivate_error(self, reference, generator_backprop=False):
        deriv_vector = self._activation_function.derivate(self.activation_levels)
        if generator_backprop:
            return deriv_vector * self.error_gen.derivate(reference, self.output)
        else:
            return deriv_vector * self.error.derivate(reference, self.output)