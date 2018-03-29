import numpy as np


class GanGame:
    """
    Class of en GAN game, i.e two network learning together with the GAN theory
    """

    ##
    # @param      discriminator   The discriminator (will be a Network object)
    # @param      generator       The generator (will be a Network object)
    # @param      learning_ratio  The learning ratio between discrimator and generator
    ##
    def __init__(self, discriminator, learning_set, learning_fun, generator, eta_gen, eta_disc, disc_learning_ratio=1, gen_learning_ratio=1, disc_fake_learning_ratio=0, gen_learning_ratio_alone=0):
        self.generator = generator
        self.discriminator = discriminator
        self.learning_set = learning_set
        self.set_size = len(learning_set)
        self.learning_fun = learning_fun
        self.eta_gen = eta_gen
        self.eta_disc = eta_disc
        self.gen_learning_ratio = gen_learning_ratio
        self.disc_learning_ratio = disc_learning_ratio
        self.disc_fake_learning_ratio = disc_fake_learning_ratio
        self.gen_learning_ratio_alone = gen_learning_ratio_alone


    ##
    # @brief      Execute a movement of the game, learning of dicriminator, then the generator
    #
    # @return     Return either the discriminator trust the fake image or not at the moment.
    ##
    def playAndLearn(self):
        for i in range(self.disc_learning_ratio):
            self.discriminatorLearningReal()

        for j in range(self.gen_learning_ratio):
            fake_image = self.generatorLearning()
            self.discriminatorLearningVirt(fake_image)

        for k in range(self.disc_fake_learning_ratio):
            fake_image, noise = self.generateImage()
            self.discriminatorLearningVirt(fake_image, True)

        for j in range(self.gen_learning_ratio_alone):
            self.generatorLearning()

        return 0

    def testDiscriminatorLearning(self, n):
        real_trust = []
        fake_trust = []
        for i in range(n):
            real_item = self.learning_set[np.random.randint(self.set_size)]
            real_score = self.testTruth(real_item)
            real_trust.append(real_score)

        for j in range(n):
            fake_image, noise = self.generateImage()
            fake_score = self.testTruth(fake_image)
            fake_trust.append(fake_score)

        return np.mean(real_trust), np.mean(fake_trust), np.std(real_trust), np.std(fake_trust)

    ##
    # @brief      discriminator learning what is real image
    ##
    def discriminatorLearningReal(self):
        real_item = self.learning_set[np.random.randint(self.set_size)]  # generate  a random item from the set
        # expected_output = self.learning_fun.out(real_item)
        self.discriminator.compute(real_item)
        self.discriminator.backprop(self.eta_disc, real_item, 1)  # expected output = 1 pour le moment

        return 0

    ##
    # @brief      discriminator learning what is fake image
    ##
    def discriminatorLearningVirt(self, fake_image, alone=False):
        if alone:
            self.discriminator.compute(fake_image)
        self.discriminator.backprop(self.eta_disc, fake_image, 0)

        return 0

    ##
    # @brief      initiate backprop for generator
    #
    # @param      fooled  result between 0 and 1 if G fooled D or not
    #
    # @comment    The cost function will be initialize with the network.
    ##
    def generatorLearning(self):
        fake_image, noise = self.generateImage()
        fooled = self.testTruth(fake_image)
        disc_error_influence = self.discriminator.backprop(self.eta_gen, fake_image, fooled, False, True)
        self.generator.backprop(self.eta_gen, noise, disc_error_influence, self.discriminator.layers_list[0].weights)

        return fake_image

    def generateImage(self):
        noise = self.generateNoise()
        image = self.generator.compute(noise)
        return image, noise

    def generateNoise(self):
        n = self.generator.layers_neuron_count[0]
        return 2*np.random.random(n)-1

    ##
    # @brief      Give belief of discrimator about the image given
    ##
    def testTruth(self, image):
        return self.discriminator.compute(image)
