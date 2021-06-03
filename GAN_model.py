import tensorflow as tf
from keras.layers import BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.data import Dataset
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from global_vars import LATENT_NOISE_SIZE, GENERATOR_LR, CRITIC_LR


class GAN:
    def __init__(self, **kwargs):
        self._input_size = kwargs.get('input_size', None)

        if self._input_size is None:
            raise ValueError('input_size must not be None')

        self._latent_noise_size = kwargs.get('latent_size', LATENT_NOISE_SIZE)
        self._generator_activation_function = kwargs.get('generator_activation_function', 'relu')
        self._discriminator_activation_function = kwargs.get('discriminator_activation_function', 'relu')
        self._generator_lr = kwargs.get('generator_lr', GENERATOR_LR)
        self._critic_lr = kwargs.get('critic_lr', CRITIC_LR)

        self.generator = self._build_generator()
        self.critic = self._build_critic()
        self.gan = Sequential([self.generator, self.critic])
        self._compile()

    def _build_generator(self):
        generator = Sequential()
        generator.add(Dense(128, activation=self._generator_activation_function, kernel_initializer='he_uniform', input_dim=self._latent_noise_size))
        generator.add(BatchNormalization())
        generator.add(Dense(256, activation=self._generator_activation_function, kernel_initializer='he_uniform'))
        generator.add(BatchNormalization())
        generator.add(Dense(self._input_size))

        return generator

    def _build_critic(self):
        discriminator = Sequential()
        discriminator.add(Dense(128, activation=self._discriminator_activation_function, kernel_initializer='he_uniform', input_dim=self._input_size))
        discriminator.add(BatchNormalization())
        discriminator.add(Dense(64, activation=self._discriminator_activation_function, kernel_initializer='he_uniform'))
        discriminator.add(BatchNormalization())
        discriminator.add(Dense(1, activation='sigmoid'))

        return discriminator

    def _compile(self):
        generator_optimizer = RMSprop(learning_rate=self._generator_lr)
        critic_optimizer = RMSprop(learning_rate=self._critic_lr)

        self.critic.compile(loss="binary_crossentropy", optimizer=critic_optimizer, metrics=['accuracy'])
        self.critic.trainable = False

        self.gan.compile(loss="binary_crossentropy", optimizer=generator_optimizer)

    def train_gan(self, dataset: Dataset, batch_size: int, n_epochs: int):
        generator, discriminator = self.gan.layers
        for epoch in range(n_epochs):
            for X_batch in dataset:
                X_batch = tf.cast(X_batch, tf.float32)

                # phase 1 - training the discriminator
                noise = tf.random.normal(shape=[batch_size, self._latent_noise_size])
                generated_samples = generator(noise)

                X_fake_and_real = tf.concat([generated_samples, X_batch], axis=0)
                y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)

                discriminator.trainable = True
                discriminator.train_on_batch(X_fake_and_real, y1)

                # phase 2 - training the generator
                noise = tf.random.normal(shape=[batch_size, self._latent_noise_size])
                y2 = tf.constant([[1.]] * batch_size)

                discriminator.trainable = False
                self.gan.train_on_batch(noise, y2)

    def generate_samples(self, samples_num: int):
        noise = tf.random.normal(shape=[samples_num, self._latent_noise_size])
        generated_samples = self.generator(noise, training=False)

        return generated_samples


class WGAN_GP(GAN):
    def __init__(self, **kwargs):
        GAN.__init__(self, kwargs)

        #self._critic_training_steps =

