import matplotlib.pyplot as plt
import os
from time import gmtime, strftime
import numpy as np


class GanPlot:

    def __init__(self, name, number_to_draw):
        self._name = name
        self.number_to_draw = number_to_draw

    def save(self, out_vector, img_name, step_number, D_x, D_G_z):  # out_vector est le vecteur colonne avant reshape
        # create directory if it doesn't exist
        if not os.path.exists(self._name):
            os.mkdir(self._name)
        if not os.path.exists(self._name + '/Images'):
            os.mkdir(self._name + '/Images')
        image = np.reshape(out_vector, [28, 28])
        plt.imshow(image, cmap='Greys')
        plt.colorbar()  # devrait donner une correspondance entre le niveau de gris et la valeur 'réele'
        plt.title('Tentative du GAN de générer un ' + str(self.number_to_draw) + ' après '+ str(step_number)+' parties')
        plt.suptitle('D(x) = ' + str(D_x)+ ', D(G(z)) = ' + str(D_G_z))
        save_date = strftime('%Y-%m-%d-%H%M%S', gmtime())
        plt.savefig(self._name + '/Images/' + save_date + '_imagede_' + img_name + '.png')  # sauvgarde de l'image
        plt.close()
        
    def plot(self, out_vector, step_number, D_x, D_G_z):
        image = np.reshape(out_vector, [28, 28])
        plt.imshow(image, cmap='Greys')
        plt.colorbar()  # devrait donner une correspondance entre le niveau de gris et la valeur 'réele'
        plt.title('Tentative du GAN de générer un ' + str(self.number_to_draw) + ' après '+ str(step_number)+' parties')
        plt.plot()
        plt.close()