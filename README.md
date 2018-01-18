# Salamandre-Code
Projet de réseau de neurones s'intéressant à la banque de données MNIST en Python.
Le projet à vocation à devenir un socle pour l'application des théories de génération de données par GAN

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