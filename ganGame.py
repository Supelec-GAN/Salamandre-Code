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
    def __init__(self, discriminator, generator, learning_ratio=1):
        self.generator = generator
        self.discriminator = discriminator
        self.learning_ratio = learning_ratio

    ##
    # @brief      Execute a movement of the game, learning of dicriminator, then the generator
    #
    # @return     Return either the discriminator trust the fake image or not at the moment.
    ##
    def playAndLearn(self):
        for i in range(self.learning_ratio):
            self.discriminatorLearning()
        noise = self.generateNoise()
        fake_image = self.generateImage(noise)

        fooled = self.testTruth(fake_image)

        generatorLearning(fooled, noise)
        discriminatorLearningNegative(fake_image)
        return fooled

    ##
    # @brief      discriminator learning what is real image
    ##
    def discriminatorLearningPositive(self):
        self.discriminator.learn()

    ##
    # @brief      discriminator learning what is fake image
    ##
    def discriminatorLearningNegative(self, fake_image, noise):
        self.discriminator.net.backprop(self.discriminator.eta, fake_image, noise)

    ##
    # @brief      initiate backprop for generator
    #
    # @param      fooled  result between 0 and 1 if G fooled D or not
    #
    # @comment    The cost function will be initialize with the network.
    ##
    def generatorLearning(self, fooled, noise):
        self.generator.backprop(eta, noise, fooled)

    def generateImage(self, noise):
        Image = self.generator.net.compute(noise)
        return Image

    def generateNoise(self):
        n = generator.layers_neuron_count[0]
        return np.random(n)

    ##
    # @brief      Give belief of discrimator about the image given
    ##
    def testTruth(self, image):
        return self.discriminator.net.compute(image)