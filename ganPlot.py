import matplotlib.pyplot as plt
import os
from time import gmtime, strftime
import numpy as np
from matplotlib.gridspec import GridSpec


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
        plt.close

    def plot_noise(self, out_vector, step_number, D_x, D_G_z):
        image = np.reshape(out_vector, [int(np.sqrt(len(out_vector))), int(np.sqrt(len(out_vector)))])
        plt.imshow(image, cmap='Greys')
        plt.colorbar()  # devrait donner une correspondance entre le niveau de gris et la valeur 'réele'
        plt.title('bruit pour ' + str(self.number_to_draw) + ' après '+ str(step_number)+' parties')
        plt.plot()
        plt.close()

    def plot_scores(self, D_x, D_G_z):
        plt.plot(D_x, label='D(z)')
        plt.plot(D_G_z, label='D(G(z))')
        plt.ylabel('Score')
        plt.xlabel('Parties')
        plt.title('Evolution du score du Discriminant et du générateur')

    def plot_network_state(self, state):
        plt.close
        params, coefs = state
        layers_size = params[0]
        n = len(layers_size)
        fig = plt.figure()
        L = 20
        gs = GridSpec(L, n-1)
        for i in range(1, n):
            max_w = np.max(coefs[i-1][0])
            min_w = np.min(coefs[i-1][0])
            ax = plt.subplot(gs[0:L-2, i-1])
            self.plot_weight(coefs[i-1][0], layers_size[i-1], layers_size[i])
            ax.set_title('Matrice de poids : ' + str(layers_size[i-1]) + ',' + str(layers_size[i]) + ' Max : ' + str(max_w) + " Min :" + str(min_w))

            max_b = np.max(coefs[i-1][1])
            min_b = np.min(coefs[i-1][1])
            ax = plt.subplot(gs[L-1,i-1])
            self.plot_bias(coefs[i-1][1])
            ax.set_title('Biais Max : ' + str(max_b) + "Min :" + str(min_b))
        plt.show()


    def plot_weight(self, weights, input_size, output_size):
        weights = np.reshape(weights, [input_size, output_size])
        plt.imshow(weights, cmap='Greys', aspect='auto')

    def plot_bias(self, bias):
        image = np.reshape(bias, ([1, len(bias)]))
        plt.imshow(image, cmap='Greys', aspect='auto')