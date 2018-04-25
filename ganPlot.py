import matplotlib.pyplot as plt
import os
from time import gmtime, strftime
import numpy as np
from matplotlib.gridspec import GridSpec


class GanPlot:

    def __init__(self, name, numbers_to_draw):
        self._name = "ReleveExp/" + name
        self.numbers_to_draw = numbers_to_draw

    def save(self, out_vector, img_name, step_number, D_x, D_G_z):  # out_vector est le vecteur
        # colonne avant reshape
        # create directory if it doesn't exist
        if not os.path.exists(self._name + '/Images'):
            os.makedirs(self._name + '/Images')
        image = np.reshape(out_vector, [28, 28])
        plt.imshow(image, cmap='Greys')
        plt.colorbar()  # devrait donner une correspondance entre le niveau de gris et la valeur
        # 'réelle'
        plt.title('Tentative du GAN de générer un ' + str(self.numbers_to_draw) + ' après ' +
                  str(step_number)+' parties')
        plt.suptitle('D(x) = ' + str(D_x) + ', D(G(z)) = ' + str(D_G_z))
        save_date = strftime('%Y-%m-%d-%H%M%S', gmtime())
        plt.savefig(self._name + '/Images/' + save_date + '_imagede_' + img_name + '.png')
        # sauvgarde de l'image
        plt.close()

    def plot(self, out_vector, step_number, D_x, D_G_z):
        image = np.reshape(out_vector, [28, 28])
        plt.imshow(image, cmap='Greys')
        plt.colorbar()  # devrait donner une correspondance entre le niveau de gris et la valeur
        # 'réelle'
        plt.title('Tentative du GAN de générer un ' + str(self.numbers_to_draw) + ' après ' +
                  str(step_number)+' parties')
        plt.plot()
        plt.close()

    def save_multiple_output(self, out_vectors, img_name, step_number, D_x, D_G_z):  # out_vector
        # est le vecteur colonne avant reshape
        # create directory if it doesn't exist
        if not os.path.exists(self._name + '/Images'):
            os.makedirs(self._name + '/Images')
        n = len(out_vectors)
        high = (n-1)//5+1
        gs = GridSpec(high, min(n, 5))
        fig = plt.figure()
        for i in range(n):
            image = np.reshape(out_vectors[i][:,0], [28, 28])
            sub = fig.add_subplot(gs[i//5, i % 5])
            sub.imshow(image, cmap='Greys')
            sub.set_xticks([])
            sub.set_yticks([])
            # plt.colorbar()  # devrait donner une correspondance entre le niveau de gris et
            # la valeur 'réelle'
        fig.suptitle('Tentative du GAN de générer un ' + str(self.numbers_to_draw) + ' après ' +
                     str(step_number)+' parties' + '\n D(x) = ' + str(D_x) + ', D(G(z)) = ' +
                     str(D_G_z))
        save_date = strftime('%Y-%m-%d-%H%M%S', gmtime())
        plt.savefig(self._name + '/Images/' + save_date + '_imagede_' + img_name + '.png')
        # sauvgarde de l'image
        plt.close()

    def plot_multiple_output(self, out_vectors, step_number, D_x, D_G_z):
        n = len(out_vectors)
        high = (n-1)//5+1
        gs = GridSpec(high, min(n, 5))
        fig = plt.figure()
        for i in range(n):
            image = np.reshape(out_vectors[i], [28, 28])
            sub = fig.add_subplot(gs[i//5, i % 5])
            sub.imshow(image, cmap='Greys')
            sub.set_xticks([])
            sub.set_yticks([])
            # plt.colorbar()  # devrait donner une correspondance entre le niveau de gris et la
            # valeur 'réelle'
        fig.suptitle('Tentative du GAN de générer un ' + str(self.numbers_to_draw) + ' après '+ str(step_number)+' parties')
        plt.plot()
        plt.close()

    def plot_noise(self, out_vector, step_number, D_x, D_G_z):
        image = np.reshape(out_vector, [int(np.sqrt(len(out_vector))), int(np.sqrt(len(out_vector)))])
        plt.imshow(image, cmap='Greys')
        plt.colorbar()  # devrait donner une correspondance entre le niveau de gris et la valeur
        # 'réelle'
        plt.title('bruit pour ' + str(self.numbers_to_draw) + ' après ' + str(step_number) +
                  ' parties')
        plt.plot()
        plt.close()

    def plot_network_state(self, state):
        plt.close()
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
            ax = plt.subplot(gs[L-1, i-1])
            self.plot_bias(coefs[i-1][1])
            ax.set_title('Biais Max : ' + str(max_b) + "Min :" + str(min_b))
        plt.show()

    def save_plot_network_state(self, state):
        if not os.path.exists(self._name):
            os.mkdir(self._name)
        if not os.path.exists(self._name + '/Images'):
            os.mkdir(self._name + '/Images')

        plt.close()
        params, coefs = state
        layers_size = params[0]
        n = len(layers_size)
        fig = plt.figure()
        L = 20
        gs = GridSpec(L, n)
        for i in range(1, n):
            max_w = np.max(coefs[i-1][0])
            min_w = np.min(coefs[i-1][0])
            ax = plt.subplot(gs[0:L-3, i-1])
            self.plot_weight(coefs[i-1][0], layers_size[i-1], layers_size[i])
            ax.set_title('Matrice de poids : [' + str(layers_size[i-1]) + ',' + str(layers_size[i]) + '] \n Max : ' + str(max_w)[:4] + " | Min :" + str(min_w)[:4],fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            max_b = np.max(coefs[i-1][1])
            min_b = np.min(coefs[i-1][1])
            ax = plt.subplot(gs[L-1, i-1])
            self.plot_bias(coefs[i-1][1])
            ax.set_title('Biais Max : ' + str(max_b)[:4] + "\n Biais Min :" + str(min_b)[:4],
                         fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        ax = plt.subplot(gs[:, -1])
        final_bias = np.reshape(coefs[-1][1], [28, 28])
        ax.imshow(final_bias, cmap='Greys')    
        ax.set_title('Dernier Biais')
        ax.set_xticks([])
        ax.set_yticks([])

        save_date = strftime('%Y-%m-%d-%H%M%S', gmtime())
        plt.savefig(self._name + '/Images/' + save_date + 'Reseau_final' + '.png',
                    bbox_inches='tight', dpi=300)  # sauvegarde de l'image
        plt.close()

    def plot_weight(self, weights, input_size, output_size):
        weights = np.reshape(weights[:, 0:input_size], [input_size, output_size])
        plt.imshow(weights, cmap='Greys', aspect='auto')

    @staticmethod
    def plot_bias(bias):
        image = np.reshape(bias, ([1, len(bias)]))
        plt.imshow(image, cmap='Greys', aspect='auto')

    @staticmethod
    def plot_courbes(param, param_desc_gen, param_desc_disc, data_real, data_fake):
        plt.close()
        gs = GridSpec(1, 11)
        fig = plt.figure()
        images = slice(0, param['play_number']//param['test_period'], (param['play_number']//param['nb_images_during_learning'])//param['test_period'])

        ax_D_x = fig.add_subplot(gs[0, 0:10])
        ax_D_x.autoscale(axis='x')
        ax_D_x.plot(data_real, '.-', label='D(x)', markevery=images)

        ax_D_x.plot(data_fake, '.-', label='D(G(z))', markevery=images)
        ax_D_x.set_xlabel("Nombre de parties (X" + str(param['test_period']) + ")")

        ax_D_x.legend(loc=0, bbox_to_anchor=(1.4, 0.2))
        ax_D_x.set_title("Réponse du Discriminateur à des images du set et à des images de "
                         "synthèse")

        info = fig.add_subplot(gs[0, 10])

        info.set_xticks([])
        info.set_yticks([])
        info.axis('off')
        info.text(0.01, 0.95, 'Tentative pour ' + str(param['numbers_to_draw']), fontsize=16)

        info.text(0.01, 0.87, 'Formes des réseau', fontsize=12)
        info.text(0.01, 0.83, 'Forme du générateur : ' + str(param['generator_network_layers']),
                  fontsize=8)
        info.text(0.01, 0.79, 'Bruit du générateur : ' + str(param['noise_layers_size']),
                  fontsize=8)
        info.text(0.01, 0.75, 'Forme du discriminateur : ' + str(param['disc_network_layers']),
                  fontsize=8)

        info.text(0.01, 0.65, "Ratios d'apprentissages", fontsize=12)
        info.text(0.01, 0.61, 'Ratio D image du set : ' +
                  str(param['disc_learning_ratio']), fontsize=8)
        info.text(0.01, 0.57, 'Ratio G et D même image de synthèse : ' +
                  str(param['gen_learning_ratio']), fontsize=8)
        info.text(0.01, 0.53, 'Ratio D image de synthèse : ' +
                  str(param['disc_fake_learning_ratio']), fontsize=8)
        info.text(0.01, 0.49, 'Ratio G image de synthèse : ' +
                  str(param['gen_learning_ratio_alone']), fontsize=8)
             
        info.text(0.01, 0.39, "Infos courbe", fontsize=12)
        info.text(0.01, 0.35, 'Nombre de partie : ' + str(param['play_number']),
                  fontsize=8)
        info.text(0.01, 0.31, 'Test toutes les ' + str(param['test_period']) + ' parties',
                  fontsize=8)
        info.text(0.01, 0.27, 'Moyenne sur ' + str(param['lissage_test']) + ' samples par test',
                  fontsize=8)
        info.text(0.01, 0.23, "Echantillons d'images toutes les  " +
                  str(param['play_number']//param['nb_images_during_learning']) + " parties",
                  fontsize=8)
        plt.show()

    def save_courbes(self, param, param_desc_gen, param_desc_disc, data_real, data_fake):
        if not os.path.exists(self._name):
            os.mkdir(self._name)
        if not os.path.exists(self._name + '/Images'):
            os.mkdir(self._name + '/Images')

        plt.close()
        fig = plt.figure()

        gs = GridSpec(1, 11)
        images = slice(0, param['play_number']//param['test_period'],
                       (param['play_number']//param['nb_images_during_learning']) //
                       param['test_period'])

        ax_D_x = fig.add_subplot(gs[0, 0:10])
        ax_D_x.autoscale(axis='x')
        ax_D_x.plot(data_real, '.-', label='D(x)', markevery=images)

        ax_D_x.plot(data_fake, '.-', label='D(G(z))', markevery=images)
        ax_D_x.set_xlabel("Nombre de parties (X" + str(param['test_period']) + ")")

        ax_D_x.legend(loc=0, bbox_to_anchor=(1.3, 0.1))
        ax_D_x.set_title("Réponse du Discriminateur à des images du set et à des images de "
                         "synthèse")

        info = fig.add_subplot(gs[0, 10])

        info.set_xticks([])
        info.set_yticks([])
        info.axis('off')
        info.text(0.01, 0.95, 'Tentative pour ' + str(param['numbers_to_draw']), fontsize=16)

        info.text(0.01, 0.87, 'Formes des réseau', fontsize=12)
        info.text(0.01, 0.83, 'Forme du générateur : ' + str(param['generator_network_layers']),
                  fontsize=8)
        info.text(0.01, 0.79, 'Bruit du générateur : ' + str(param['noise_layers_size']),
                  fontsize=8)
        info.text(0.01, 0.75, 'Forme du discriminateur : ' + str(param['disc_network_layers']),
                  fontsize=8)

        info.text(0.01, 0.61, 'Ratio D image du set : ' +
                  str(param['disc_learning_ratio']), fontsize=8)
        info.text(0.01, 0.57, 'Ratio G et D même image de synthèse : ' +
                  str(param['gen_learning_ratio']), fontsize=8)
        info.text(0.01, 0.53, 'Ratio D image de synthèse : ' +
                  str(param['disc_fake_learning_ratio']), fontsize=8)
        info.text(0.01, 0.49, 'Ratio G image de synthèse : ' +
                  str(param['gen_learning_ratio_alone']), fontsize=8)
        info.text(0.01, 0.45, 'Eta générateur : ' +
                  str(param_desc_gen['eta_gen']), fontsize=8)
        info.text(0.01, 0.41, 'Eta discriminateur : ' +
                  str(param_desc_disc['eta_disc']), fontsize=8)

        info.text(0.01, 0.31, "Infos courbe", fontsize=12)
        info.text(0.01, 0.27, 'Nombre de partie : ' + str(param['play_number']),
                  fontsize=8)
        info.text(0.01, 0.23, 'Test toutes les ' + str(param['test_period']) + ' parties',
                  fontsize=8)
        info.text(0.01, 0.19, 'Moyenne sur ' + str(param['lissage_test']) + ' samples par test',
                  fontsize=8)
        info.text(0.01, 0.15, "Echantillons d'images toutes les  " +
                  str(param['play_number']//param['nb_images_during_learning']) + " parties",
                  fontsize=8)
        
        save_date = strftime('%Y-%m-%d-%H%M%S', gmtime())
        plt.savefig(self._name + '/Images/' + save_date + 'Courbes' + '.png',
                    bbox_inches='tight', dpi=300)  # sauvegarde de l'image
        plt.close()
