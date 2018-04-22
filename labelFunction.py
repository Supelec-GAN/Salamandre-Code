import numpy as np


class LabelFunction:

    def __init__(self, *args):
        pass

    def label(self, *args):
        pass


class XorTest(LabelFunction):

    def __init__(self, mini=0, maxi=1):
        """
        Class for exclusive-or test.
        :param mini: Value returned when XOR is false
        :param maxi: Value returned when XOR is true
        """
        self.mini = mini
        self.maxi = maxi

    def label(self, x, y):
        return self.maxi*((x > 0) ^ (y > 0)) - self.mini*(1-(x > 0) ^ (y > 0))


class MnistTest(LabelFunction):

    def __init__(self, set_labels):
        """
        Class for mnist (or cifar10) test.

        :param set_labels:
        """
        self._set_labels = set_labels

    def label(self, x):
        if type(x) in [int, np.int32, np.int64]:
            r = np.zeros(10)
            r[self._set_labels[x]] = 1
            r = np.reshape(r, (10, 1))
        else:
            n = len(x)
            r = np.zeros((10, n))
            for i in range(n):
                r[self._set_labels[x[i]]][i] = 1
        return r


class MnistGanTest(LabelFunction):

    def __init__(self):
        """
        Class for mnist gan test.
        """
        pass

    def label(self, x):
        return 1
