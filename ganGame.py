"""
Aucun test, c'est une sorte de pseudo-code, une methode agile askip,
on remplit au fur et à mesure pour que ça marche.
"""

import numpy as np

class GanGame:
    """
    Class of en GAN game, i.e two network learning together with the GAN theorie
    """

    def __init__(self, discriminator, generator, learning_ratio=1):
        self.generator = generator
        self.discriminator = discriminator
        self.learning_ratio = learning_ratio

    def playAndLearn(self):
        for i in range(self.learning_ratio):
            self.discriminatorLearning()
        noise = self.generateNoise()
        fake_image = self.generateImage(noise)

        fooled = self.testTruth(fake_image)

        self.generatorLearning(fooled)
        return fooled

    def discriminatorLearning(self):
        self.discriminator.learn()   
        pass

    def generatorLearning(self):
        self.generator.learn()
        pass

    ##
    # @brief      { function_description }
    #
    # @param      noise  The noise
    #
    # @return     { description_of_the_return_value }
    ##
    def generateImage(self, noise):
        Image = self.generator.net.compute(noise)
        return Image

    def generateNoise(self):
        random = np.random()
        self.noiseFunction.out()

    def testTruth(self, image):
        return self.discriminator.net.compute(image)