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


class MnistTest(Function):

    def __init__(self, set_labels):
        self._set_labels = set_labels

    def out(self, x):
        r = np.zeros(10)
        r[self._set_labels[x]] = 1
        r = np.reshape(r, (10, 1))
        return r


class Norm2(Function):

    def __init__(self):
        pass

    def out(self, reference, x):
        return np.linalg.norm(x - reference)

    def derivate(self, reference, x):
        return -2 * (reference - x)

##
## @brief      Class for non saturant heuritic for GAN. 
## Parameter x is useless, only for matching the neuronLayer class
## (cf init_derivate_error function)
##
class NonSatHeuritic(Function):

    def __init__(self):
        pass

    def out(self, reference, useless):
        return -0.5*np.log(reference)

    def derivate(self, reference, useless):
        return -0.5/reference