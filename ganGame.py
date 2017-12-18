"""
Aucun test, c'est une sorte de pseudo-code, une methode agile askip,
on remplit au fur et à mesure pour que ça marche.
"""


class GanGame:
    """
    Class of en GAN game, i.e two network learning together with the GAN theorie
    """

    ##
    ## @brief      Constructs the object.
    ##
    ## @param      self            The object
    ## @param      discriminator   The discriminator (will be a Engine object)
    ## @param      generator       The generator (will be a only a network)
    ## @param      learning_ratio  The learning ratio between discrimator and generator
    ##
    def __init__(self, discriminator, generator, learning_ratio=1):
        self.generator = generator
        self.discriminator = discriminator
        self.learning_ratio = learning_ratio

    def playAndLearn(self):
        for i in range(learning_ratio):
            self.discriminatorLearningPositive()
        noise = self.generateNoise()
        fake_image = self.generateImage(noise)

        fooled = testTruth(fake_image)

        generatorLearning(fooled, noise)
        discriminatorLearningNegative(fake_image)
        return fooled

    def discriminatorLearningPositive():
        self.discriminator.learn()   
        pass

    def discriminatorLearningNegative(fake_image, noise):
        self.discriminator.net.backprop(self.discriminator.eta, fake_image, noise)
    ##
    ## @brief      initiate backprop for generator
    ##
    ## @param      fooled  result between 0 and 1 if G fooled D or not
    ##
    ## @comment    The cost function will be initialize with the network.
    def generatorLearning(fooled, noise):
        self.generator.backprop(eta, noise, fooled)

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

    def generateNoise():  # pas forcément adapté de l'avoir ici
        random = np.random()
        self.noiseFunction.out()

    def testTruth(image):
        return discriminator.net.compute(image)