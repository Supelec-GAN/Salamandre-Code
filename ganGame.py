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

    def __init__(self, discriminator, learning_set, learning_fun, generator,
                 disc_learning_ratio=1, gen_learning_ratio=1, disc_fake_learning_ratio=0, 
                 gen_learning_ratio_alone=0, batch_size=0):
        self.generator = generator
        self.discriminator = discriminator
        self.learning_set = learning_set
        self.set_size = len(learning_set)
        self.learning_fun = learning_fun
        self.gen_learning_ratio = gen_learning_ratio
        self.disc_learning_ratio = disc_learning_ratio
        self.disc_fake_learning_ratio = disc_fake_learning_ratio
        self.gen_learning_ratio_alone = gen_learning_ratio_alone
        self.batch_size = batch_size


    ##
    # @brief      Execute a movement of the game, learning of dicriminator, then the generator
    #
    # @return     Return either the discriminator trust the fake image or not at the moment.
    ##
    def playAndLearn(self):
        for i in range(self.disc_learning_ratio):
            self.discriminatorLearningReal()
            # print("discRealLearnCheck", i)

        for j in range(self.gen_learning_ratio):
            fake_images = self.generatorLearning()
            self.discriminatorLearningVirt(fake_images)

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
            real_item = np.transpose([self.learning_set[np.random.randint(self.set_size)] for i in range(self.batch_size)])
            real_score = self.testTruth(real_item)
            real_trust.append(real_score)

        for j in range(n):
            fake_images, noise = self.generateImage()
            noises = [noise]*self.batch_size
            fake_score = self.testTruth(np.transpose(fake_images))
            fake_trust.append(fake_score)

        return np.mean(real_trust), np.mean(fake_trust), np.std(real_trust), np.std(fake_trust)

    ##
    # @brief      discriminator learning what is real image
    ##
    def discriminatorLearningReal(self):
        real_items = np.transpose([self.learning_set[np.random.randint(self.set_size)] for i in range(self.batch_size)])  # generate  a random item from the set
        # expected_output = self.learning_fun.out(real_item)
        self.discriminator.compute(real_items)
        self.discriminator.backprop(real_items, np.ones((self.batch_size, 1))) # expected output = 1 pour le moment

        return 0

    ##
    # @brief      discriminator learning what is fake image
    ##
    def discriminatorLearningVirt(self, fake_images, alone=False):
        if alone:
            self.discriminator.compute(fake_images)
        self.discriminator.backprop(fake_images, np.zeros((self.batch_size, 1)))

        return 0

    ##
    # @brief      initiate backprop for generator
    #
    # @param      fooled  result between 0 and 1 if G fooled D or not
    #
    # @comment    The cost function will be initialize with the network.
    ##
    def generatorLearning(self):
        fake_images, noises = self.generateImage()
        # real_items = [self.learning_set[np.random.randint(self.set_size)] for i in range(self.batch_size)]
        # batch = fake_images.concatenate(real_items)
        batch = np.transpose(fake_images)
        fooled = self.testTruth(fake_images)

        disc_error_influence = self.discriminator.backprop(fake_images, fooled, False, True)
        self.generator.backprop(noises, disc_error_influence, self.discriminator.layers_list[0].weights)

        return fake_images

    def generateImage(self):
        noises = self.generateNoise()
        images = self.generator.compute(noises)
        return images, noises

    def generateNoise(self):
        n = self.generator.layers_neuron_count[0]
        noises = 2*np.random.random((n, self.batch_size))-1
        return noises

    ##
    # @brief      Give belief of discrimator about the image given
    ##
    def testTruth(self, image):
        return self.discriminator.compute(image)
