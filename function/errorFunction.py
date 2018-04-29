import numpy as np


class ErrorFunction:

    def __init__(self):
        """
        Classe abtraite pour les fonctions d'erreurs des réseaux
        """
        pass

    def out(self, *args):
        """
        Valeur de l'erreur

        :param args: Paramètres nécessaires pour calculer l'erreur (généralement output et
        reference)
        :return: Valeur de l'erreur
        """
        raise NotImplementedError

    def derivate(self, *args):
        """
        Valeur de la dérivée de l'erreur

        :param args: Paramètres nécessaures pour calculer l'erreur (généralement output et
        reference)
        :return: Valeur de la dérivée
        """
        raise NotImplementedError

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
        return str(self) + '()'

    def __repr__(self):
        return 'ErrorFunction'


class Norm2(ErrorFunction):

    def __init__(self):
        """
        Classe calculant la norme 2
        """
        super(Norm2, self).__init__()

    def out(self, reference, x):
        return np.linalg.norm(x - reference, axis=0)

    def derivate(self, reference, x):
        return -2 * (reference - x)

    def __repr__(self):
        return 'Norm2'


class NonSatHeuristic(ErrorFunction):

    def __init__(self):
        """
        Class for non saturant heuristic for GAN
        """
        super(NonSatHeuristic, self).__init__()

    def out(self, output):
        return -0.5*np.log(output)

    def derivate(self, output):
        return -0.5/output

    def __repr__(self):
        return 'NonSatHeuristic'


class CostFunction(ErrorFunction):

    def __init__(self):
        super(CostFunction, self).__init__()

    def out(self, reference, output):
        if reference == 1:
            return -0.5*np.log(output)
        else:
            return -0.5*np.log(1 - output)

    ##
    # Reference est 1 si on donne une vrai image, 0 si c'est une image virtuelle
    ##
    def derivate(self, reference, output):
        if reference == 1:
            return -0.5/output
        else:
            return +0.5/(1-output)

    def __repr__(self):
        return 'CostFunction'


class CrossEntropy(ErrorFunction):

    def __init__(self):
        super(CrossEntropy, self).__init__()

    def out(self, reference, output):
        return -(reference*np.log(output) + (1-reference)*np.log(1-output))

    def derivate(self, reference, output):
        return -(reference/output - (1-reference)/(1-output))

    def __repr__(self):
        return 'CrossEntropy'

# class GeneratorError(ErrorFunction):

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
