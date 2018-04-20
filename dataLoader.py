import numpy as np
from mnist import MNIST
import pickle


def load_mnist(path):
    mndata = MNIST(path)
    training_images, training_labels = mndata.load_training()
    training_images = np.array(training_images)/256  # Normalisation de l'image (pixel entre 0 et 1)

    testing_images, testing_labels = mndata.load_testing()
    testing_images = np.array(testing_images)/256  # Normalisation de l'image (pixel entre 0 et 1)

    return training_images, training_labels, testing_images, testing_labels


def load_cifar10(path):
    training_images = np.empty((0, 3072), dtype=np.float64)
    training_labels = np.empty(0, dtype=np.int32)
    for nb in range(1,6):
        with open(path + '/' + 'data_batch_' + str(nb), 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        training_images = np.vstack((training_images, batch[b'data']))
        training_labels = np.hstack((training_labels, batch[b'labels']))
    training_labels = np.array(training_labels)

    with open(path + '/' + 'test_batch', 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    testing_images = batch[b'data']
    testing_labels = batch[b'labels']

    return training_images, training_labels, testing_images, testing_labels


def load_data(path, dataset):
    if dataset in ['MNIST', 'Mnist', 'mnist']:
        return load_mnist(path)
    elif dataset in ['CIFAR10', 'Cifar10', 'cifar10']:
        return load_cifar10(path)
    else:
        raise Exception('Wrong dataset')
