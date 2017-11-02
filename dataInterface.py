from time import gmtime, strftime
import numpy as np
import os


##
# @brief      Class for data interface.
#
# @param      name Name of folder used to save the date
#
class DataInterface:
    def __init__(self, name='XOR'):
        self._name = name

    ##
    # @brief      save numpy array data into the folder self._name
    #
    # @param      data_name   descricption of the data(error, weights matrix, )
    # @param      data_param  Parameters of network and run of the dataset
    #
    # @return     No return, filename is name\YYYY-MM-DD-HHmmSS_data_name.csv
    def save(self, data, data_name, data_param=np.array([10, 1000, 100, 0.01]), param_description='parallel_learnings, test_period, iterations, eta'):
        data_param_str = self.save_param(data_param)
        print(data_param_str)
        save_date = strftime('%Y-%m-%d-%H%M%S', gmtime())

        # create directory if it doesn't exist
        if not os.path.exists(self._name):
            os.mkdir(self._name)

        return np.savetxt(self._name + '\\' + save_date + '_' + data_name + '.csv', data, delimiter=",", header=data_param_str, footer=param_description)

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
        params = np.fromstring(param_str, dtype=int, sep=' ')
        return params
