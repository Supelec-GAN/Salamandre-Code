import numpy as np
from fonction import Sigmoid, MnistTest, Norm2, NonSatHeuristic
from math import sqrt
from dataInterface import DataInterface


class NeuronLayer:
    """Classe permettant de créer une couche de neurones"""

    def __init__(self, activation_function, error_function, input_size=1, output_size=1, error_function_gen=NonSatHeuristic()):
        # Matrice de dimension q*p avec le nombre de sortie et p le nombre d'entrée
        self._input_size = input_size
        self._output_size = output_size
        self.weights = np.transpose(np.random.randn(input_size, output_size))
        self._bias = np.zeros((output_size, 1))            # Vecteur colonne
        self._activation_function = activation_function
        self.activation_levels = np.zeros((output_size, 1))  # Vecteur colonne
        self.output = np.zeros((output_size, 1))             # Vecteur colonne
        self.error = error_function
        self.error_gen = error_function_gen

        self.update_weights_value = np.zeros((output_size, input_size))
        self.update_bias_value = np.zeros((output_size, 1))

        self.weights_gradients_sum = np.zeros((output_size, input_size))
        self.bias_gradients_sum = np.zeros((output_size, 1))
        self.weights_moment = np.zeros((output_size, input_size))
        self.bias_moment = np.zeros((output_size, 1))
        self.weights_eta = np.zeros((output_size, input_size))          #need meilleur nom
        self.bias_eta = np.zeros((output_size, 1))                      #need meilleur nom

        data_interface = DataInterface()
        param_liste = data_interface.read_conf('config_algo_descente.ini', 'Parametres de descente')  # Lecture du fichier de config
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
    #
    def backprop(self, out_influence, eta, input_layer, update=True):
        self.eta = eta
        weight_influence = self.calculate_weight_influence(
            input_layer, out_influence)
        bias_influence = self.calculate_bias_influence(out_influence)
        if update:
            self.update_momentum(bias_influence, weight_influence)
            self.update_weights(eta, weight_influence)
            self.update_bias(eta, bias_influence)
            return self.weights
        else:
            return self.weights - eta * weight_influence 

    def update_momentum(self, bias_influence, weight_influence):

        if self.algo_utilise == "Gradient":
            self.update_weights_value = self.momentum*self.update_weights_value - self.eta*weight_influence
            self.update_bias_value = self.momentum*self.update_bias_value - self.eta * bias_influence

        elif self.algo_utilise == "Adagrad":
            for i in range(len(weight_influence)):
                for j in range(len(weight_influence[0])):
                    self.weights_gradients_sum[i][j] = self.weights_gradients_sum[i][j] + weight_influence[i][j]**2
                    self.update_weights_value[i][j] = self.momentum*self.update_weights_value[i][j] - self.eta*weight_influence[i][j]/(sqrt(self.weights_gradients_sum[i][j])+self.epsilon)
            for i in range(len(bias_influence)):
                for j in range(len(bias_influence[0])):
                    self.bias_gradients_sum[i][j] = self.bias_gradients_sum[i][j] + bias_influence[i][j]**2
                    self.update_bias_value[i][j] = self.momentum*self.update_bias_value[i][j] - self.eta*weight_influence[i][j]/(sqrt(self.bias_gradients_sum[i][j])+self.epsilon)

        elif self.algo_utilise == "RMSProp":

            if self.moment == 2:
                for i in range(len(weight_influence)):
                    for j in range(len(weight_influence[0])):
                        self.weights_gradients_sum[i][j] = self.gamma*self.weights_gradients_sum[i][j] + (1 - self.gamma)*weight_influence[i][j]**2
                        self.update_weights_value[i][j] = self.momentum*self.update_weights_value[i][j] - self.eta*weight_influence[i][j]/(sqrt(self.weights_gradients_sum[i][j])+ self.epsilon)
                for i in range(len(bias_influence)):
                    for j in range(len(bias_influence[0])):
                        self.bias_gradients_sum[i][j]= self.gamma*self.bias_gradients_sum[i][j] + (1-self.gamma)*bias_influence[i][j]**2
                        self.update_bias_value[i][j] = self.momentum*self.update_bias_value[i][j] - self.eta*bias_influence[i][j]/(sqrt(self.bias_gradients_sum[i][j])+self.epsilon)

            if self.moment == 1:
                for i in range(len(weight_influence)):
                    for j in range(len(weight_influence[0])):
                        self.weights_gradients_sum[i][j] = self.gamma * self.weights_gradients_sum[i][j] + (1 - self.gamma) * weight_influence[i][j] ** 2
                        self.weights_moment[i][j] = self.gamma * self.weights_moment[i][j] + (1- self.gamma) * weight_influence[i][j]
                        self.update_weights_value[i][j] = self.momentum*self.update_weights_value[i][j] - self.eta*weight_influence[i][j]/sqrt(self.weights_gradients_sum[i][j] - self.weights_moment[i][j]**2 + self.epsilon)
                for i in range(len(bias_influence)):
                    for j in range(len(bias_influence[0])):
                        self.bias_gradients_sum[i][j] = self.gamma * self.bias_gradients_sum[i][j] + (1 - self.gamma) * bias_influence[i][j] ** 2
                        self.bias_moment[i][j] = self.gamma * self.bias_moment[i][j] + (1 - self.gamma) * bias_influence[i][j]
                        self.update_bias_value[i][j] = self.momentum*self.update_bias_value[i][j] - self.eta*bias_influence[i][j]/sqrt(self.bias_gradients_sum[i][j] - self.bias_moment[i][j]**2 + self.epsilon)

        elif self.algo_utilise == "Adadelta":

            if self.moment == 2:
                for i in range(len(weight_influence)):
                    for j in range(len(weight_influence[0])):
                        self.weights_gradients_sum[i][j] = self.gamma * self.weights_gradients_sum[i][j] + (1 - self.gamma) * weight_influence[i][j] ** 2
                        self.weights_eta[i][j] = self.gamma * self.weights_eta[i][j] + (1 - self.gamma) * self.update_weights_value[i][j]**2
                        self.update_weights_value[i][j] = self.momentum*self.update_weights_value[i][j] - sqrt(self.weights_eta[i][j] + self.epsilon)*weight_influence[i][j]/(sqrt(self.weights_gradients_sum[i][j])+ self.epsilon)
                for i in range(len(bias_influence)):
                    for j in range(len(bias_influence[0])):
                        self.bias_gradients_sum[i][j] = self.gamma * self.bias_gradients_sum[i][j] + (1 - self.gamma) * bias_influence[i][j] ** 2
                        self.bias_eta[i][j] = self.gamma * self.bias_eta[i][j] + (1 - self.gamma) * self.update_bias_value[i][j] ** 2
                        self.update_bias_value[i][j] = self.momentum*self.update_bias_value[i][j] - sqrt(self.bias_eta[i][j] + self.epsilon)*bias_influence[i][j]/(sqrt(self.bias_gradients_sum[i][j])+self.epsilon)

            if self.moment == 1:
                for i in range(len(weight_influence)):
                    for j in range(len(weight_influence[0])):
                        self.weights_gradients_sum[i][j] = self.gamma * self.weights_gradients_sum[i][j] + (1 - self.gamma) * weight_influence[i][j] ** 2
                        self.weights_eta[i][j] = self.gamma * self.weights_eta[i][j] + (1 - self.gamma) * self.update_weights_value[i][j] ** 2
                        self.weights_moment[i][j] = self.gamma * self.weights_moment[i][j] + (1 - self.gamma) * weight_influence[i][j]
                        self.update_weights_value[i][j] = self.momentum*self.update_weights_value[i][j] - sqrt(self.weights_eta[i][j] + self.epsilon)*weight_influence[i][j]/sqrt(self.weights_gradients_sum[i][j] - self.weights_moment[i][j]**2 + self.epsilon)
                for i in range(len(bias_influence)):
                    for j in range(len(bias_influence[0])):
                        self.bias_gradients_sum[i][j] = self.gamma * self.bias_gradients_sum[i][j] + (1 - self.gamma) * bias_influence[i][j] ** 2
                        self.bias_eta[i][j] = self.gamma * self.bias_eta[i][j] + (1 - self.gamma) * self.update_bias_value[i][j] ** 2
                        self.bias_moment[i][j] = self.gamma * self.bias_moment[i][j] + (1 - self.gamma) * bias_influence[i][j]
                        self.update_bias_value[i][j] = self.momentum*self.update_bias_value[i][j] - sqrt(self.bias_eta[i][j] + self.epsilon)*bias_influence[i][j]/sqrt(self.bias_gradients_sum[i][j] - self.bias_moment[i][j]**2 + self.epsilon)

        elif self.algo_utilise == "Adam":

            self.instant += 1
            for i in range(len(weight_influence)):
                for j in range(len(weight_influence[0])):
                    self.weights_gradients_sum[i][j] = self.gamma * self.weights_gradients_sum[i][j] + (1 - self.gamma) * weight_influence[i][j] ** 2
                    self.weights_moment[i][j] = self.gamma * self.weights_moment[i][j] + (1 - self.gamma) * weight_influence[i][j]
                    self.update_weights_value[i][j] = self.momentum * self.update_weights_value[i][j] - self.alpha*self.weights_moment[i][j]/(1 - self.gamma_1**self.instant)*1/(sqrt(self.weights_gradients_sum[i][j]/(1 - self.gamma_2**self.instant)) + self.epsilon)
            for i in range(len(bias_influence)):
                for j in range(len(bias_influence[0])):
                    self.bias_gradients_sum[i][j] = self.gamma * self.bias_gradients_sum[i][j] + (1 - self.gamma) * bias_influence[i][j] ** 2
                    self.bias_moment[i][j] = self.gamma * self.bias_moment[i][j] + (1 - self.gamma) * bias_influence[i][j]
                    self.update_bias_value[i][j] = self.momentum * self.update_bias_value[i][j] - self.alpha*self.bias_moment[i][j]/(1-self.gamma_1**self.instant)*1/(sqrt(self.bias_gradients_sum[i][j]/(1 - self.gamma_2**self.instant))+ self.epsilon)

    def update_weights(self, eta, weight_influence):
        # self.update_weights_value = momentum*self.update_weights_value - eta * weight_influence
        self.weights = self.weights + self.update_weights_value

    def update_bias(self, eta, bias_influence):
        # self.update_bias_value = momentum * self.update_bias_value + eta * bias_influence
        self._bias = self._bias + self.update_bias_value

    ##
    # @brief      Calculates the weight influence.
    #
    # @param      input_layer    input of the last compute
    # @param      out_influence  influence of output on the error
    #
    # @return     vecteur of same dimension than weights.
    ##
    def calculate_weight_influence(self, input_layer, out_influence):
        return np.dot(out_influence, np.transpose(input_layer))

    ##
    # @brief      Calculates the bias influence (which is out_influence)
    ##
    def calculate_bias_influence(self, out_influence):
        return out_influence

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

    def derivate_error(self, reference, generator_backprop=False):
        deriv_vector = self._activation_function.derivate(self.activation_levels)
        if generator_backprop:
            return deriv_vector * self.error_gen.derivate(reference, self.output)
        else:
            return deriv_vector * self.error.derivate(reference, self.output)


##
# @brief      Class for layer with noisy input added to inputs.
##
class NoisyLayer(NeuronLayer):
    def __init__(self, activation_function, error_function, input_size=1, output_size=1, noise_size=0, error_function_gen=NonSatHeuristic()):
        super(NoisyLayer, self).__init__(activation_function, error_function, input_size, output_size, error_function_gen)
        self._noise_size = noise_size
        self.weights = np.transpose(np.random.randn(input_size+noise_size, output_size))
        self.noise_input = np.zeros((noise_size, 1))

        self.weights_gradients_sum = np.zeros((output_size, input_size + noise_size))
        self.update_weights_value = np.zeros((output_size, input_size + noise_size))
        self.weights_moment = np.zeros((output_size, input_size + noise_size))
        self.weights_eta = np.zeros((output_size, input_size + noise_size))          #need meilleur nom
        
    ##
    # Compute très légèrement différent, on concatene un vecteur de bruits à l'input si nécéssaire
    ##
    def compute(self, inputs):
        if self._noise_size != 0:  # nécessaire car np.zeros( (0,1)) est un objet chelou
            self.noise_input = np.random.randn(self._noise_size, 1)
            inputs = np.concatenate([inputs, self.noise_input])
        self.activation_levels = np.dot(self.weights, inputs) - self._bias
        self.output = self._activation_function.out(self.activation_levels)
        return self.output

    ##
    # backptop très légèrement différent, on retropropage en considérant le vecteur bruit,
    # mais sans renvoyer son influence à la couche précédente
    ##
    def backprop(self, out_influence, eta, input_layer, update=True):
        if self._noise_size != 1:
            input_layer = np.concatenate([input_layer, self.noise_input])
        weight_influence = self.calculate_weight_influence(
            input_layer, out_influence)
        bias_influence = self.calculate_bias_influence(out_influence)
        if update:
            self.update_momentum(bias_influence, weight_influence)
            self.update_weights(eta, weight_influence)
            self.update_bias(eta, bias_influence)
            return self.weights[:, 0:self._input_size]  # On extrait les poids concernant les vrais inputs (le bruit n'a pas besoin d'influer sur les couches d'avant)
        else:
            return (self.weights - eta * weight_influence)[:, 0:self._input_size]
