import matplotlib.pyplot as plt
import os
from time import gmtime, strftime
import numpy as np


class GanPlot:

    def __init__(self, name, number_to_draw, nb_plays):
        self._name = name
        self.number_to_draw = number_to_draw
        self.nb_plays = nb_plays

    def save(self, out_vector):  # out_vector est le vecteur colonne avant reshape
        # create directory if it doesn't exist
        if not os.path.exists(self._name):
            os.mkdir(self._name)
        image = np.reshape(out_vector, [28, 28])
        plt.imshow(image, cmap='Greys',
                   interpolation='bicubic')
        plt.colorbar()  # devrait donner une correspondance entre le niveau de gris et la valeur 'réele'
        plt.title('Tentative du GAN de générer un ' + str(self.number_to_draw) + 'après '+ str(self.nb_plays)+'parties')
        save_date = strftime('%Y-%m-%d-%H%M%S', gmtime())
        filename = str(save_date) + '.png'
        plt.savefig(filename)

    def plot(self, out_vector):
        image = np.reshape(out_vector, [28, 28])
        plt.imshow(image, cmap='Greys',
                   interpolation='bicubic')
        plt.colorbar()  # devrait donner une correspondance entre le niveau de gris et la valeur 'réele'
        plt.title('Tentative du GAN de générer un ' + str(self.number_to_draw) + 'après '+ str(self.nb_plays)+'parties')
        plt.plot()
