"""
Il manque la gestion du bruit
"""

import numpy as np

class GanGame:
    """
    Class of en GAN game, i.e two network learning together with the GAN theory
    """

    ##
    # @param      discriminator   The discriminator (will be a Engine object)
    # @param      generator       The generator (will be a only a network object)
    # @param      learning_ratio  The learning ratio between discrimator and generator
    ##
    def __init__(self, discriminator, generator, eta_gen, learning_ratio=1):
        self.generator = generator
        self.discriminator = discriminator
        self.learning_ratio = learning_ratio
        self.eta_gen = eta_gen

    ##
    # @brief      Execute a movement of the game, learning of dicriminator, then the generator
    #
    # @return     Return either the discriminator trust the fake image or not at the moment.
    ##
    def playAndLearn(self):
        for i in range(self.learning_ratio):
            self.discriminatorLearningPositive()
        fake_image, noise = self.generateImage()

        fooled = self.testTruth(fake_image)

        self.generatorLearning(fooled, noise)
        self.discriminatorLearningNegative(fake_image)
        return fooled

    ##
    # @brief      discriminator learning what is real image
    ##
    def discriminatorLearningPositive(self):
        self.discriminator.learn()

    ##
    # @brief      discriminator learning what is fake image
    ##
    def discriminatorLearningNegative(self, fake_image):
        self.discriminator.net.backprop(self.discriminator.eta, fake_image, 0)

    ##
    # @brief      initiate backprop for generator
    #
    # @param      fooled  result between 0 and 1 if G fooled D or not
    #
    # @comment    The cost function will be initialize with the network.
    ##
    def generatorLearning(self, fooled, noise):
        self.generator.backprop(self.eta_gen, noise, fooled)

    def generateImage(self):
        noise = self.generateNoise()
        image = self.generator.compute(noise)
        return image, noise

    def generateNoise(self):
        n = self.generator.layers_neuron_count[0]
        return np.random.random(n)

    ##
    # @brief      Give belief of discrimator about the image given
    ##
    def testTruth(self, image):
        return self.discriminator.net.compute(image)