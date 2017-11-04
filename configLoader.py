from configparser import ConfigParser
import json
from fonction import *


class ConfigLoader:
    def __init__(self, configfile):
        self.cfg = ConfigParser()
        self.cfg.read(configfile)

    def read(self, type='Mnist'):
        file = self.cfg[type]['file']
        training_size = self.cfg[type].getint('training_size')
        testing_size = self.cfg[type].getint('testing_size')

        learning_iterations = self.cfg[type]getint('learning_iterations')
        test_period = self.cfg[type]getint('test_period')
        randomize_learning_set = self.cfg[type]getboolean('randomize_learning_set')

        activation_funs = self.cfg[type]['activation_funs']
        error_fun = self.cfg[type]['error_fun']
        network_layers = json.load(self.cfg[type]['network_layers'])

        eta = self.cfg[type]['eta']

        SaveFolder = self.cfg[type]['SaveFolder']

        return file, training_size, testing_size, learning_iterations, test_period, randomize_learning_set, activation_funs, error_fun, network_layers, eta, SaveFolder

    def explode_list(fun):
        crochet = array_fun.split('[')[1].split(']')[0]
        listes_str = crochet.split(', ')
        fun_list = []
        arg_list = []
        for string in listes_str:
            fun, arg = explode_function(string)
            fun_list.append(fun)
            arg_list.append(arg)
        return fun_list, arg_list

    def explode_function(string):
        xplode = string.split('(')
        fun = xplode[0]
        arg = int(xplode[1].split(')')[0].split(', '))
        return fun, arg
    def getListFunction(array_fun):


    def getFunction(array_fun):
        fun_list, arg_list = self.explode_list(array_fun)
        final_list = []
        for i in range(len(fun_list)):
            fun = fun_list[i]
            arg = arg_list[i]
            if fun == 'Sigmoid':
                final_list.append(Sigmoid(arg[0]))
            elif fun == 'Tanh':
                final_list.append(Tanh(arg[0], arg[1]))
            elif fun = 'XorTest'

