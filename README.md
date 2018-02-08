# Salamandre-Code
Projet de réseau de neurones s'intéressant tout d'abord à la banque de données MNIST en Python.
Le projet à vocation à devenir un socle pour l'application des théories de génération de données par GAN.

-- 
Utilisation du Réseau de Neurones pour MNIST.

À l'aide du script mnistMain.py importer un fichier de configuration.
Ce dernier est un config.ini.

Il suffit d'écrire dans la section [Mnist] les paramètres comme si on instanciait les objets python.

read_conf(self, filename='config.ini', param='Mnist')

--
Utilisation du Réseau de Neurones pour le GAN sur MNIST.

À l'aide du script ganMnistMain.py importer un fichier de configuration.
Ce dernier est un config.ini.

Il suffit d'écrire dans la section [GanMnist] les paramètres comme si on instanciait les objets
python.

read_conf(self, filename='config.ini', param='GanMnist')

Exemple de fichier de configuration :

```
[GanMnist]
file = './data'  # Fichier sources des chiffres MNIST

numbers_to_draw = [5, 4] # Liste des nombres que l'on veut générer
 
disc_learning_ratio = 1 # Ratio d'apprentissage du discriminateur avec une image de la base MNIST
disc_fake_learning_ratio = 0 # Ratio d'apprentissage du discriminateur avec une image issu du générateur

disc_activation_funs = [Sigmoid(0.1), Sigmoid(0.1), Sigmoid(0.1)] # fonction d'activation des couches du discriminateur
disc_error_fun = CostFunction() # Fonction d'erreur du discriminateur pour l'apprentissage du discriminateur
disc_network_layers = [784, 10, 1] # taille des couches 

eta_disc = 1 # pas d'apprentissage pour le discriminateur

training_fun = MnistGanTest # fonction de réponse pour l'apprentissage 



generator_network_layers = [100, 300, 784] # Taille des couches 
noise_layers_size = [0, 800, 0]  # taille du bruit ajouté sur 
generator_activation_funs = [Sigmoid(0.1), Sigmoid(0.1), Sigmoid(0.1)] # fonction d'activation des couches du generateur

gen_learning_ratio = 1 # Ratio d'apprentissage du générateur avec discriminateur
gen_learning_ratio_alone = 0 # Ratio d'apprentissage du générateur seul
eta_gen = 1 # pas d'apprentissage pour le générateur


nb_images_during_learning = 100 # Nombre d'images exportés au cours de l'apprentissage
final_images = 10 # Nombre d'images exportés après l'apprentissage (donc avec n bruits pour le même réseau)

play_number = 1000000 # Nombre de parties

save_folder = '2couchebruit800_4et5' # Dossier de sauvegarde des courbes D(x) et D(G(z))
test_period = 10000 # Période des tests pour obtenir les courbes (attention petit nombre ralentit le process)
lissage_test = 1 # Nombre de test pour une série (attention c'est n compute du générateur de 2n compute du discrimateur) 

```
