import numpy as np


class LabelFunction:

    def __init__(self, *args):
        """
        General class for label functions

        :param args: The dataset labels informations
        """
        pass

    def label(self, *args):
        """
        Method that returns the appropriate label

        :param args: The label wanted
        :return: The label associated to args
        """
        raise NotImplementedError

    def vectorize(self):
        self.label = np.vectorize(self.label)


class XorTest(LabelFunction):

    def __init__(self, mini=0, maxi=1):
        """
        Class for exclusive-or test.
        :param mini: Value returned when XOR is false
        :param maxi: Value returned when XOR is true
        """
        super(XorTest, self).__init__()
        self.mini = mini
        self.maxi = maxi

    def label(self, x, y):
        return self.maxi*((x > 0) ^ (y > 0)) - self.mini*(1-(x > 0) ^ (y > 0))


class MnistTest(LabelFunction):

    def __init__(self, set_labels, class_count=10):
        """
        Class for mnist (or cifar10) test.

        :param set_labels: The numeric labels for the set
        :param class_count: The number of class in the set
        """
        super(MnistTest, self).__init__()
        self._set_labels = set_labels
        self._class_count = class_count

    def label(self, x):
        if type(x) in [int, np.int32, np.int64]:
            expected_output = np.zeros(self._class_count)
            expected_output[self._set_labels[x]] = 1
            return expected_output.reshape((1, -1))
        else:
            n = len(x)
            expected_output = np.zeros((n, self._class_count))
            for i in range(n):
                expected_output[i][self._set_labels[x[i]]] = 1
            return expected_output




class MnistGanTest(LabelFunction):

    def __init__(self):
        """
        Class for mnist gan test.
        """
        super(MnistGanTest, self).__init__()

    def label(self, x):
        return 1
