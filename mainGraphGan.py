import numpy as np
from brain.network import Network
from dataInterface import DataInterface
from errorGraphs import ErrorGraphs
import matplotlib.pyplot as plt
from ganPlot import GanPlot


data_interface = DataInterface('C://Users//Froux//Documents//Projet_Long//Data//GanMnist//Courbes')
param_real, data_real = data_interface.load("2018-01-31-204909_discriminator_real_score.csv")
param_fake, data_fake = data_interface.load("2018-01-31-204909_discriminator_fake_score.csv")

numbers_to_draw = param_fake['numbers_to_draw']

save_folder = param_fake['save_folder']

gan_plot = GanPlot(save_folder, numbers_to_draw)

gan_plot.save_courbes(param_fake, data_real, data_fake)

