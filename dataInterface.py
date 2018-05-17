from time import gmtime, strftime
import matplotlib.pyplot as plt
import os
from configparser import ConfigParser
from function.labelFunction import *
from function.errorFunction import *
from function.activationFunction import *


class DataInterface:

    def __init__(self, name='Mnist'):
        """
        Class for data interface

        :param name: Name of the folder used to save the data
        """
        self._name = "ReleveExp/" + name

    def rename(self, name):
        self._name = "ReleveExp/" + name

    def save(self, data, data_name, data_param='dictionnary of parameters'):
        """
        Save numpy array data into the folder self._name. Filename is
        name\YYYY-MM-DD-HHmmSS_data_name.csv

        :param data: Data to save
        :param data_name: Description of the data (error, weights matrix, ...)
        :param data_param: Parameters of network and run of the dataset
        :return: None
        """
        save_date = strftime('%Y-%m-%d-%H%M%S', gmtime())

        # create directory if it doesn't exist
        if not os.path.exists(self._name):
            os.makedirs(self._name)

        return np.savetxt(self._name + '/' + save_date + '_' + data_name + '.csv', data,
                          delimiter=",", header=data_param)

    @staticmethod
    def save_param(data_param):
        """
        Transform an np.array into a string to save params

        :param data_param: An array of params
        :return: A string of params
        """
        return np.array_str(data_param).split('[')[1].split(']')[0]

    def load(self, filename):
        """
        Load data from a file

        :param filename: File containing the data
        :return: An np.array with parameters of acquisition and a dataset
        """
        params = self.load_param(filename)
        data = np.loadtxt(self._name + '/' + filename, delimiter=',')
        return params, data

    def load_old(self, filename):
        """
        Load data from a file (deprecated)

        :param filename: File containing the data
        :return: Two np.array with parameters of acquisition and a dataset
        """
        param1, param2 = self.load_param_old(filename)
        data = np.loadtxt(self._name + '/' + filename, delimiter=',')
        return param1, param2, data

    def load_param_old(self, filename):
        """
        Read the parameters line of csv file (deprecated)

        :param filename: The file to read
        :return: Two np.array of parameters
        """
        file = open(self._name + '/' + filename)

        first = file.readline()
        second = file.readline()
        lines = file.readlines()
        last = lines[len(lines) - 1]
        param_first = first.split('# ')[1].split('\n')[0] + second.split('# ')[1].split('\n')[0]
        param_last = last.split('# ')[1].split('\n')[0]

        params1 = param_first.split(' ')
        params2 = (param_last.split('+')[-1].split('Network(')[-1].split('], '))
        params2[0] = eval(params2[0] + ']')
        params2[1] = eval(params2[1])

        params1 = list(filter(None, params1))
        for i in range(len(params1)):
            params1[i] = eval(params1[i])
        return params1, params2

    def load_param(self, filename):
        """
        Read the parameters from a file

        :param filename: The file to read
        :return: A dictionary of parameters
        """
        file = open(self._name + '/' + filename)
        first = file.readline()
        param_str = first.split('# ')[1].split('\n')[0]

        params = eval(param_str)
        param_dict = {}
        for opt in params:
            param_dict[opt] = eval(params[opt])
        return param_dict

    @staticmethod
    def read_conf(filename='config.ini', param='Mnist'):
        cfg = ConfigParser()
        cfg.read(filename)
        options = cfg.options(param)
        param_dict = {}

        for opt in options:
            param_dict[opt] = eval(cfg[param][opt])

        return param_dict

    @staticmethod
    def save_conf(filename='config.ini', param='Mnist'):
        cfg = ConfigParser()
        cfg.read(filename)
        options = cfg.options(param)
        param_dict_str = {}

        for opt in options:
            param_dict_str[opt] = cfg[param][opt]
        return str(param_dict_str)

    @staticmethod
    def load_conf(param_dict):
        for key in param_dict.keys():
            param_dict[key] = eval(param_dict[key])

        return param_dict

    def save_img_black(self, image, img_name, x_size=28, y_size=28):

        save_date = strftime('%Y-%m-%d-%H%M%S', gmtime())
        # create directory if it doesn't exist
        if not os.path.exists(self._name + '/Images'):
            os.makedirs(self._name + '/Images')

        image = np.reshape(image, [x_size, y_size])
        plt.imshow(image, cmap='Greys',  interpolation='nearest')
        # sauvegarde de l'image
        plt.savefig(self._name + '/Images/' + save_date + '_imagede_' + img_name + '.png')

    @staticmethod
    def extract_param(param_liste, i):
        param = dict()
        for key, value in param_liste.items():
            n = len(value)
            if i < n:
                param[key] = value[i]
            else:
                param[key] = value[-1]
        return param
