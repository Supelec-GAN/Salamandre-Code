"""
Aucun test, c'est une sorte de pseudo-code, une methode agile askip,
on remplit au fur et à mesure pour que ça marche.
"""


class GanGame:
    """
    Class of en GAN game, i.e two network learning together with the GAN theorie
    """

    def __init__(self, discriminator, generator, learning_ratio=1):
        self.generator = generator
        self.discriminator = discriminator
        self.learning_ratio = learning_ratio

    def playAndLearn(self):
        for i in range(learning_ratio):
            self.discriminatorLearning()
        noise = self.generateNoise()
        fake_image = self.generateImage(noise)

        fooled = testTruth(fake_image)

        generatorLearning(fooled)
        return fooled

    def discriminatorLearning():
        self.discriminator.learn()   
        pass

    def generatorLearning():
        pass

    ##
    # @brief      { function_description }
    #
    # @param      noise  The noise
    #
    # @return     { description_of_the_return_value }
    ##
    def generateImage(noise):
        Image = generator.net.compute(noise)
        return Image

    def generateNoise():
        random = np.random()
        self.noiseFunction.out()

    def testTruth(image):
        return discriminator.net.compute(image)