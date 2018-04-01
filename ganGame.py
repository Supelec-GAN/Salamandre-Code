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
    def __init__(self, discriminator, learning_set, learning_fun, generator, eta_gen, eta_disc,
                 disc_learning_ratio=1, gen_learning_ratio=1, disc_fake_learning_ratio=0,
                 gen_learning_ratio_alone=0, batch_size=1):
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
        self.batch_size = batch_size

    ##
    # @brief      Execute a movement of the game, learning of dicriminator, then the generator
    #
    # @return     Return either the discriminator trust the fake image or not at the moment.
    ##
    def play_and_learn(self):
        for i in range(self.disc_learning_ratio):
            self.discriminator_learning_real()

        for j in range(self.gen_learning_ratio):
            fake_images = self.generator_learning()
            self.discriminator_learning_virt(fake_images)

        for k in range(self.disc_fake_learning_ratio):
            fake_image, noise = self.generate_image()
            self.discriminator_learning_virt(fake_image, True)

        for j in range(self.gen_learning_ratio_alone):
            self.generator_learning()

        return 0

    def test_discriminator_learning(self, n):
        real_trust = []
        fake_trust = []
        for i in range(n):
            real_item = np.transpose([self.learning_set[np.random.randint(self.set_size)]
                                      for i in range(self.batch_size)])
            real_score = self.test_truth(real_item)
            real_trust.append(real_score)

        for j in range(n):
            fake_images, noise = self.generate_image()
            print('fakereal ', np.shape(fake_images))
            noises = [noise]*self.batch_size
            fake_score = self.test_truth(np.transpose(fake_images))
            fake_trust.append(fake_score)

        return np.mean(real_trust), np.mean(fake_trust), np.std(real_trust), np.std(fake_trust)

    ##
    # @brief      discriminator learning what is real image
    ##
    def discriminator_learning_real(self):
        real_items = np.transpose([self.learning_set[np.random.randint(self.set_size)]
                                   for i in range(self.batch_size)])
        # generate a random item from the set
        # expected_output = self.learning_fun.out(real_item)
        self.discriminator.compute(real_items)
        self.discriminator.backprop(self.eta_disc, real_items, np.ones((self.batch_size, 1)))
        # expected output = 1 pour le moment

        return 0

    ##
    # @brief      discriminator learning what is fake image
    ##
    def discriminator_learning_virt(self, fake_images, alone=False):
        if alone:
            self.discriminator.compute(fake_images)
        self.discriminator.backprop(self.eta_disc, fake_images, np.zeros((self.batch_size, 1)))

        return 0

    ##
    # @brief      initiate backprop for generator
    #
    # @param      fooled  result between 0 and 1 if G fooled D or not
    #
    # @comment    The cost function will be initialize with the network.
    ##
    def generator_learning(self):
        fake_images, noises = self.generate_image()
        # real_items = [self.learning_set[np.random.randint(self.set_size)]
        #               for i in range(self.batch_size)]
        # batch = fake_images.concatenate(real_items)
        # batch = np.transpose(fake_images)
        fooled = self.test_truth(fake_images)

        disc_error_influence = self.discriminator.backprop(self.eta_gen, fooled, False, True)
        self.generator.backprop(self.eta_gen, disc_error_influence)

        return fake_images

    def generate_image(self):
        noises = self.generate_noise()
        images = self.generator.compute(noises)
        return images, noises

    def generate_noise(self):
        n = self.generator.layers_neuron_count[0]
        noises = 2*np.random.random((n, self.batch_size))-1
        return noises

    ##
    # @brief      Give belief of discrimator about the image given
    ##
    def test_truth(self, image):
        return self.discriminator.compute(image)
