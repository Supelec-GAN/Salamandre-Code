import numpy as np
from random import randint


class GanGame:

    def __init__(self, discriminator, learning_set, learning_fun, generator,
                 disc_learning_ratio=1, gen_learning_ratio=1, disc_fake_learning_ratio=0,
                 gen_learning_ratio_alone=0, batch_size=0, image_number=20):
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

        n = self.generator.input_size
        self.noises_test = [2 * np.random.random((n, self.batch_size)) - 1 for i in range(image_number)]

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
            noises = [noise] * self.batch_size  # tjs nécessaire ?
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
        self.discriminator.backprop(np.ones((self.batch_size, 1)))
        # expected output = 1 pour le moment
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

    def generate_image_test(self):
        """
        Generates an image by inputting noise in the generator

        :return: The generated image and the noise used
        """

        images = [self.generator.compute(noise, False) for noise in self.noises_test]
        # images2 = [self.generator.compute(noise) for noise in self.noises_test]
        # return np.concatenate((images, images2), axis=0)
        return images

    def generate_noise(self):
        """
        Generates noise for the generator. This noise is uniformally distributed in [-1, 1[

        :return: The created noise
        """
        n = self.generator.input_size
        noises = 2 * np.random.random((n, self.batch_size)) - 1
        return noises

    def test_truth(self, image):
        """
        Gives belief of discriminator about the image given. Basically just a compute of
        the discriminator

        :param image: The image put to the test
        :return: The answer of the discriminator
        """
        return self.discriminator.compute(image)


class WGanGame(GanGame):
    def __init__(self, critic, learning_set, learning_fun, generator,
                 critic_learning_ratio=1, gen_learning_ratio=1,
                 batch_size=0, image_number=20):

        super(WGanGame, self).__init__(discriminator=critic,
                                       learning_set=learning_set,
                                       learning_fun=learning_fun,
                                       generator=generator,
                                       disc_learning_ratio=critic_learning_ratio,
                                       gen_learning_ratio=gen_learning_ratio,
                                       disc_fake_learning_ratio=0,
                                       gen_learning_ratio_alone=0,
                                       batch_size=batch_size,
                                       image_number=image_number)

        self.nb_batch = self.set_size // self.batch_size * 10
        print("nb_batch", self.nb_batch)
        self.batch_critic_true = np.array([np.transpose(self.learning_set[np.random.randint(self.set_size,
                                                                      size=self.batch_size)]) for i in range(self.nb_batch)])
        # self.batch_critic_false = np.array([self.generate_image() for i in range(10)])

        # print("shape du batch tru : {}, shape du batch false : {}".format(np.shape(self.batch_critic_true), np.shape(self.batch_critic_false)))
        # self.batch_critic = np.concatenate((self.batch_critic_true, self.batch_critic_false), axis=2)
        # print("shape du batch tru : {}, shape du batch false : {}, shape du batch total : {}".format(np.shape(self.batch_critic_true), np.shape(self.batch_critic_false), np.shape(self.batch_critic)))

        self.expected = np.concatenate((np.ones((1, self.batch_size)), np.zeros((1, self.batch_size))), axis=1)

    def generate_image(self):
        """
        Generates an image by inputting noise in the generator

        :return: The generated image and the noise used
        """
        noises = self.generate_noise()
        images = self.generator.compute(noises)
        return images

    def play_and_learn(self):
        """
        Execute a movement of the game, learning of discriminator, then the generator

        :return: None
        """

        for i in range(self.disc_learning_ratio):
            self.critic_learning()

        for j in range(self.gen_learning_ratio):
            fake_images = self.generator_learning()

        return 0

    def test_critic_learning(self, n):
        """
        Teste le score du critic
        :param n: Nombre de tests
        :return: (real_score, fake_score, real_std, fake_std)
        """
        scores = []
        self.discriminator.learning_batch_size = self.batch_size
        for i in range(n):
            # # generate a random item from the set
            real_items = self.batch_critic_true[randint(0, self.nb_batch - 1)]
            # print("generate image pour test_critic_learning")
            fake_items = self.generate_image()
            # # generate samples from the generator
            # print("compute du test_critic_learning")

            score_true = self.discriminator.compute(real_items)
            score_fake = - self.discriminator.compute(fake_items)
            
            batch = np.concatenate((score_true, score_fake), axis=1)
            # print("batch", batch)
            scores.append(np.mean(batch))
        self.discriminator.learning_batch_size = self.batch_size
        return np.mean(scores), np.std(scores)

    def critic_learning(self):
        """
        critic

        :return:
        """

        self.discriminator.learning_batch_size = self.batch_size * 2

        # Set the descente parameter to false, because we want to maximize the value
        self.discriminator.set_descente(desc=False)
        # real_items = np.transpose(self.learning_set[np.random.randint(self.set_size,
        #                                                               size=self.batch_size)])
        # # generate a random item from the set
        a = randint(0, self.nb_batch - 1)
        real_items = self.batch_critic_true[a]

        # print("generate image pour critic_learning")
        fake_items = self.generate_image()

        # # generate samples from the generator
        batch = np.concatenate((real_items, fake_items), axis=1)
        # print("real : shape {}, min {}, max {}, std {}".format(np.shape(real_items[:, 0]), np.amin(real_items[:, 0]), np.amax(real_items[:, 0]), np.std(real_items[:, 0])))
        # print("fake : shape {}, min {}, max {}, std {}".format(np.shape(fake_items[:, 0]), np.amin(fake_items[:, 0]), np.amax(fake_items[:, 0]), np.std(fake_items[:, 0])))
        # print("compute du critic_learning")
        score = self.discriminator.compute(batch)
        print(np.mean(score * self.expected + (self.expected - 1) * score))
        self.discriminator.backprop(self.expected)
        # expected output = 1 pour les vrais images et 0 pour les fausses images

        self.discriminator.learning_batch_size = self.batch_size

        return 0

    def generator_learning(self):
        """
        Initiate backprop for generator. The cost function will be initialized with the network

        :return: None
        """

        # Set the descente parameter to true, because we want to minimize the value
        self.discriminator.set_descente(desc=True)
        self.discriminator.learning_batch_size = self.batch_size

        fake_images = self.generate_image()
        # real_items = [self.learning_set[np.random.randint(self.set_size)]
        #               for i in range(self.batch_size)]
        # batch = fake_images.concatenate(real_items)
        # batch = np.transpose(fake_images)
        score = self.discriminator.compute(fake_images)
        # print("score", score)
        disc_error_influence = self.discriminator.backprop(score, update=False, gen_backprop=True)
        self.generator.backprop(disc_error_influence, calculate_error=False)

        return fake_images
