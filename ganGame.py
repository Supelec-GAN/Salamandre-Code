import numpy as np


class GanGame:

    def __init__(self, discriminator, learning_set, learning_fun, generator,
                 disc_learning_ratio=1, gen_learning_ratio=1, disc_fake_learning_ratio=0,
                 gen_learning_ratio_alone=0, batch_size=0, switch_odd_fake=0.05, switch_odd_real=0.05):
        """
        Class of en GAN game, i.e two network learning together with the GAN theory

        :param discriminator: The discriminator (will be a Network object)
        :param learning_set: Real dataset
        :param learning_fun: NOT USED, DELETE ?
        :param generator: The generator (will be a Network object)
        :param disc_learning_ratio:
        :param gen_learning_ratio:
        :param disc_fake_learning_ratio:
        :param gen_learning_ratio_alone:
        :param batch_size: The batch size for the learning process
        """
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
        self.switch_odd_fake = switch_odd_fake
        self.switch_odd_real = switch_odd_real

    def play_and_learn(self):
        """
        Execute a movement of the game, learning of discriminator, then the generator

        :return: None
        """
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
        """
        Teste le score du discriminant sur des données réelles et des données fausses

        :param n: Nombre de tests
        :return: (real_score, fake_score, real_std, fake_std)
        """
        real_trust = []
        fake_trust = []
        for i in range(n):
            real_item = np.transpose(self.learning_set[np.random.randint(self.set_size,
                                                                         size=self.batch_size)])
            real_score = self.test_truth(real_item)
            real_trust.append(real_score)

        for j in range(n):
            fake_images, noise = self.generate_image()
            noises = [noise]*self.batch_size  # tjs nécessaire ?
            fake_score = self.test_truth(np.transpose(fake_images))
            fake_trust.append(fake_score)

        return np.mean(real_trust), np.mean(fake_trust), np.std(real_trust), np.std(fake_trust)

    def discriminator_learning_real(self):
        """
        Discriminator learning what is real image

        :return:
        """
        real_items = np.transpose(self.learning_set[np.random.randint(self.set_size,
                                                                      size=self.batch_size)])
        # generate a random item from the set
        # expected_output = self.learning_fun.out(real_item)
        self.discriminator.compute(real_items)
        if np.random.random() < self.switch_odd_fake:
            self.discriminator.backprop(np.zeros((self.batch_size, 1)))
        else:
            self.discriminator.backprop(0.7+0.3*np.random.random((self.batch_size, 1)))
        # expected output = entre 0.7 et 1 pour le moment
        return 0

    def discriminator_learning_virt(self, fake_images, alone=False):
        """
        Discriminator learning what is fake image

        :param fake_images: The fake images created by the generator that will be given to the
        discriminator
        :param alone: If True, compute first. If False, the compute has already been done for the
        generator learning
        :return: None
        """
        if alone:
            self.discriminator.compute(fake_images)
        if np.random.random() < self.switch_odd_fake:
            self.discriminator.backprop(np.ones((self.batch_size, 1)))
        else:
            self.discriminator.backprop(np.zeros((self.batch_size, 1)))

        return 0

    def generator_learning(self):
        """
        Initiate backprop for generator. The cost function will be initialized with the network

        :return: None
        """
        fake_images, noises = self.generate_image()
        # real_items = [self.learning_set[np.random.randint(self.set_size)]
        #               for i in range(self.batch_size)]
        # batch = fake_images.concatenate(real_items)
        # batch = np.transpose(fake_images)
        fooled = self.test_truth(fake_images)

        disc_error_influence = self.discriminator.backprop(fooled, False, True)
        self.generator.backprop(disc_error_influence, calculate_error=False)

        return fake_images

    def generate_image(self):
        """
        Generates an image by inputting noise in the generator

        :return: The generated image and the noise used
        """
        noises = self.generate_noise()
        images = self.generator.compute(noises)
        return images, noises

    def generate_noise(self):
        """
        Generates noise for the generator. This noise is uniformally distributed in [-1, 1[

        :return: The created noise
        """
        n = self.generator.input_size
        noises = 2*np.random.random((n, self.batch_size))-1
        return noises

    def test_truth(self, image):
        """
        Gives belief of discriminator about the image given. Basically just a compute of
        the discriminator

        :param image: The image put to the test
        :return: The answer of the discriminator
        """
        return self.discriminator.compute(image)
