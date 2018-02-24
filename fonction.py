import numpy as np


class Function:
    """
    @brief      classe abstraite de fonction formelle
    """

    def __init__(self, delta=0.05, *args):
        self.delta = delta

    ##
    # @brief      retourne la fonction
    #
    # @param      self
    #
    # @return     une fonction de type lambda x:
    #
    def out(self, x, *args):
        return x

    ##
    # @brief      retourne la fonction dérivée
    #
    # @param      self  The object
    #
    # @return     la dérivée formelle ou avec le delta
    #
    def derivate(self, x, *args):
        return (self.out(x+self.delta)-self.out(x))/self.delta

    def save_fun(self):
        return 'Function()'


class RelU(Function):
    """
    @brief Fonction Rectified linear units formelle
    """

    def __init__(self, alpha=0):
        self.alpha = alpha

    def out(self, x):
        if x <= 0:
            return alpha*x
        else:
            return x

    def derivate(self, x):
        if x <= 0:
            return 0
        else:
            return 1
            



class Sigmoid(Function):
    """
    @brief      Classe définissant une sigmoïde formelle
    """

    def __init__(self, mu=1):
        self.mu = mu

    def out(self, x):
        return 1/(1+np.exp(-self.mu*x))

    def derivate(self, x):
        return self.mu*np.exp(self.mu*x)/(np.power(1+np.exp(self.mu*x), 2))

    def save_fun(self):
        return 'Sigmoid({})'.format(self.mu)


class SigmoidCentered(Function):
    """
    @brief      Classe définissant une sigmoïde formelle
    """

    def __init__(self, mu=1):
        self.mu = mu

    def out(self, x):
        return 2/(1+np.exp(-self.mu*x))-1

    def derivate(self, x):
        return 2*self.mu*np.exp(self.mu*x)/(np.power(1+np.exp(self.mu*x), 2))

    def save_fun(self):
        return 'SigmoidCentered({})'.format(self.mu)


class Tanh(Function):
    """
    @brief      Classe définissant une tangeante hyperbolique formelle
    """

    def __init__(self, k=1, alpha=1):
        self.k = k
        self.alpha = alpha

    def out(self, x):
        return self.k*np.tanh(self.alpha*x)

    def derivate(self, x):
        return self.k*self.alpha/(np.power(np.cosh(self.alpha*x), 2))

    def save_fun(self):
        return 'Tanh({},{})'.format(self.k, self.alpha)


##
# @brief      Class for exclusive-or test.
#
class XorTest(Function):

    ##
    # @brief      Constructs the object.
    #
    # @param      self  The object
    # @param      mini  Valeur retournée pour Xor Faux
    # @param      maxi  Valeur retournée pour Xor Vrai
    #
    def __init__(self, mini=0, maxi=1):
        self.mini = mini
        self.maxi = maxi

    def out(self, x, y):
        return self.maxi*((x > 0) ^ (y > 0)) - self.mini*(1-(x > 0) ^ (y > 0))

    def save_fun(self):
        return 'XorTest({], {})'.format(self.mini, self.maxi)


##
# @brief      Class for mnist test.
##
class MnistTest(Function):

    def __init__(self, set_labels):
        self._set_labels = set_labels

    def out(self, x):
        r = np.zeros(10)
        r[self._set_labels[x]] = 1
        r = np.reshape(r, (10, 1))
        return r

    def save_fun(self):
        return 'MnistTest({})'.format(self._set_labels)


##
# @brief      Class for mnist gan test.
##
class MnistGanTest(Function):

    def __init__(self):
        pass

    def out(self, x):
        return 1

    def save_fun(self):
        return 'MnistGanTest()'


##
# @brief      Class for normalize 2.
##
class Norm2(Function):

    def __init__(self):
        pass

    def out(self, reference, x):
        return np.linalg.norm(x - reference)

    def derivate(self, reference, x):
        return -2 * (reference - x)

    def save_fun(self):
        return 'Norm2()'


##
# @brief      Class for non saturant heuritic for GAN. 
# Parameter x is useless, only for matching the neuronLayer class
# (cf init_derivate_error function)
##
class NonSatHeuristic(Function):

    def __init__(self):
        pass

    def out(self, output, useless):
        return -0.5*np.log(output)

    def derivate(self, output, useless):
        return -0.5/output

    def save_fun(self):
        return 'NonSatHeuritic()'

class CostFunction(Function):

    def __init__(self):
        pass

    def out(self, reference, output):
        if (reference == 1):
            return -0.5*np.log(output)
        else:
            return -0.5*np.log(1 - output)

    ##
    # Reference est 1 si on donne une vrai image, 0 si c'est une image virtuelle
    ##
    def derivate(self, reference, output):
        if(reference == 1):
            return -0.5/output
        else:
            return +0.5/(1-output)

    def save_fun(self):
        return 'CostFunction()'
        

# class GeneratorError(Function):

#     def __init__(self):
#         pass

#     def out(self, reference, output):
#         return 0

#     def derivate(self, reference, output):
#         out_influence = reference[0]
#         next_weights = reference[1]
#         return np.dot(np.transpose(next_weights), out_influence)

#     def save_fun(self):
#         return 'generatorError()'
