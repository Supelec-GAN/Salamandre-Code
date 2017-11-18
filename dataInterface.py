from time import gmtime, strftime
import numpy as np
import os
from configparser import ConfigParser
from fonction import Tanh, Sigmoid, XorTest, MnistTest, Norm2


# @brief      Class for data interface.
#
# @param      name Name of folder used to save the date
#
class DataInterface:
    def __init__(self, name='Mnist'):
        self._name = name

    ##
    # @brief      save numpy array data into the folder self._name
    #
    # @param      data_name   descricption of the data(error, weights matrix, )
    # @param      data_param  Parameters of network and run of the dataset
    #
    # @return     No return, filename is name\YYYY-MM-DD-HHmmSS_data_name.csv
    def save(self, data, data_name, data_param='dictionnary of parameters'):
        save_date = strftime('%Y-%m-%d-%H%M%S', gmtime())

        # create directory if it doesn't exist
        if not os.path.exists(self._name):
            os.mkdir(self._name)

        return np.savetxt(self._name + '\\' + save_date + '_' + data_name + '.csv', data, delimiter=",", header=data_param)

    ##
    # @brief      transform np.array into string to save param
    def save_param(self, data_param):
        return np.array_str(data_param).split('[')[1].split(']')[0]

    ##
    # @brief      load data from a file
    # @param      filename  The filename
    #
    # @return     an np.array with parameters of acquisition and a dataset
    #
    def load(self, filename):
        params = self.load_param(filename)
        data = np.loadtxt(self._name + '\\' + filename, delimiter=',')
        return params, data

    ##
    # @brief      Read the parameters line of csv file
    def load_param(self, filename):
        file = open(self._name + '\\' + filename)
        first = file.readline()
        param_str = first.split('# ')[1].split('\n')[0]
        params = eval(param_str)
        return params

    def read_conf(self, filename='config.ini', param='Mnist'):
        cfg = ConfigParser()
        cfg.read(filename)
        options = cfg.options(param)
        param_dict = {}

        for opt in options:
            param_dict[opt] = eval(cfg[param][opt])

        return param_dict

    def save_conf(self, filename='config.ini', param='Mnist'):
        cfg = ConfigParser()
        cfg.read(filename)
        options = cfg.options(param)
        param_dict_str = {}

        for opt in options:
            param_dict_str[opt] = cfg[param][opt]

        return str(param_dict_str)

    def load_conf(self, param_dict):
        for key in param_dict.keys():
            param_dict[key] = eval(param_dict[key])

        return param_dict
