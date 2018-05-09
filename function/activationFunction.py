import numpy as np


class Function:

    def __init__(self, delta=0.05):
        """
        Classe abstraire de fonction formelle

        :param delta: delta utilisé pour approximer la dérivée
        """
        self.delta = delta

    def out(self, x):
        """
        Renvoie l'évaluation de la fonction au point demandé

        :param x: Point de l'évaluation
        :return: La valeur de la fonction
        """
        return x

    def derivate(self, x):
        """
        Renvoie la valeur approchée ou exacte de la dérivée en un point

        :param x: Point de l'évaluation
        :return: la dérivée formelle ou avec le delta
        """
        return (self.out(x+self.delta)-self.out(x))/self.delta

    def vectorize(self):
        """
        Vectorize the methods of a Function

        :return: None
        """
        self.out = np.vectorize(self.out)
        self.derivate = np.vectorize(self.derivate)

    def save_fun(self):
        """
        Created a string that can be evaluated to recreate the same function later

        :return: A string
        """
        return str(self)

    def __repr__(self):
        return 'Function' + '()'


class Sigmoid(Function):

    def __init__(self, mu=1):
        """
        Classe définissant une sigmoïde formelle (sortie dans ]0,1[)

        f(x) = 1 / (1 + exp(-mu * x))

        :param mu: Paramètre de la sigmoïde
        """
        super(Sigmoid, self).__init__()
        self.mu = mu

    def out(self, x):
        return 1/(1+np.exp(-self.mu*x))

    def derivate(self, x):
        return self.mu*np.exp(self.mu*x)/(np.power(1+np.exp(self.mu*x), 2))

    def __repr__(self):
        return 'Sigmoid' + '({})'.format(self.mu)


class SigmoidCentered(Function):

    def __init__(self, mu=1):
        """
        Classe définissant une sigmoïde formelle centrée en 0 (sortie dans ]-1,-[)

        f(x) = -1 + 2/(1 + exp(-mu*x))

        :param mu: Paramètre de la sigmoïde
        """
        super(SigmoidCentered, self).__init__()
        self.mu = mu

    def out(self, x):
        return 2/(1+np.exp(-self.mu*x))-1

    def derivate(self, x):
        return 2*self.mu*np.exp(self.mu*x)/(np.power(1+np.exp(self.mu*x), 2))

    def __repr__(self):
        return 'SigmoidCentered'


class Tanh(Function):

    def __init__(self, k=1, alpha=1):
        """
        Classe définissant une tangente hyberbolique formelle

        f(x) = k * tanh(alpha*x)

        :param k: Paramètre de la tangente
        :param alpha: Paramètre de la tangente
        """
        super(Tanh, self).__init__()
        self.k = k
        self.alpha = alpha

    def out(self, x):
        return self.k*np.tanh(self.alpha*x)

    def derivate(self, x):
        return self.k*self.alpha/(np.power(np.cosh(self.alpha*x), 2))

    def __repr__(self):
        return 'Tanh' + '({},{})'.format(self.k, self.alpha)


class Relu(Function):

    def __init__(self):
        """
        Classe définissant la fonction relu formelle

        f(x) = max(0,x)
        """
        super(Relu, self).__init__()

    def out(self, x):
        if x >= 0:
            return x
        else:
            return 0

    def derivate(self, x):
        if x >= 0:
            return 1
        else:
            return 0

    def __repr__(self):
        return 'Relu' + '()'


class SoftPlus(Function):

    def __init__(self):
        """
        Classe définissant la fonction softplus formelle

        f(x) = log(1+exp(x))
        """
        super(SoftPlus, self).__init__()

    def out(self, x):
        return np.log(1 + np.exp(x))

    def derivate(self, x):
        return 1 / (1 + np.exp(-x))

    def __repr__(self):
        return 'SoftPlus' + '()'
