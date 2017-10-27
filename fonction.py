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
    def out(self):
        return lambda x: x

    ##
    # @brief      retourne la fonction dérivée
    #
    # @param      self  The object
    #
    # @return     la dérivée formelle ou avec le delta
    #
    def derivate(self):
        return lambda x: (self.out(x+self.delta)-self.out(x))/self.delta


class Sigmoid(Function):
    """
    @brief      Classe définissant une sigmoïde formelle
    """

    def __init__(self, mu=1):
        self.mu = mu

    def out(self):
        return lambda x: 1/(1+np.exp(-self.mu*x))

    def derivate(self):
        return lambda x: self.mu*np.exp(self.mu*x)/(np.power(1+np.exp(self.mu*x), 2))


class Tanh(Function):
    """
    @brief      Classe définissant une tangeante hyperbolique formelle
    """

    def __init__(self, k=1, alpha=1):
        self.k = k
        self.alpha = alpha

    def out(self):
        return lambda x: self.k*np.tanh(self.alpha*x)

    def derivate(self):
        return lambda x: self.k*self.alpha/(np.power(np.cosh(self.alpha*x), 2))


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

    def out(self):
        return lambda x, y: self.maxi*((x > 0) ^ (y > 0)) - self.mini*(1-(x > 0) ^ (y > 0))
