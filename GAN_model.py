import tensorflow as tf
from keras.layers import BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.data import Dataset
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from global_vars import LATENT_NOISE_SIZE, GENERATOR_LR, CRITIC_LR, CHECKPOINT_DIR, CHECKPOINT_PREFIX


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
        self._generator_optimizer = RMSprop(learning_rate=self._generator_lr)
        self._critic_optimizer = RMSprop(learning_rate=self._critic_lr)

        self.critic.compile(loss="binary_crossentropy", optimizer=self._critic_optimizer, metrics=['accuracy'])
        self.critic.trainable = False

        self.gan.compile(loss="binary_crossentropy", optimizer=self._generator_optimizer)

    def train_step(self, X_batch, batch_size: int):
        generator, critic = self.gan.layers

        # phase 1 - training the discriminator
        noise = tf.random.normal(shape=[batch_size, self._latent_noise_size])
        generated_samples = generator(noise)

        X_fake_and_real = tf.concat([generated_samples, X_batch], axis=0)
        y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)

        critic.trainable = True
        c_loss, c_acc = critic.train_on_batch(X_fake_and_real, y1)

        # phase 2 - training the generator
        noise = tf.random.normal(shape=[batch_size, self._latent_noise_size])
        y2 = tf.constant([[1.]] * batch_size)

        critic.trainable = False
        g_loss = self.gan.train_on_batch(noise, y2)

        return c_loss, c_acc, g_loss

    def train_gan(self, dataset: Dataset,
                  batch_size: int,
                  n_epochs: int,
                  n_eval: int = 15,
                  n_checkpoint: int = 10,
                  checkpoint_dir: str = CHECKPOINT_DIR):

        generator, critic = self.gan.layers

        # configure checkpoint to save the critic and generator models during the training process
        checkpoint = tf.train.Checkpoint(generator_optimizer=self._generator_optimizer,
                                         critic_optimizer=self._critic_optimizer,
                                         generator=generator,
                                         critic=critic)

        c_loss_per_batch = []
        c_loss_per_epoch = []
        g_loss_per_batch = []
        g_loss_per_epoch = []
        c_acc_per_batch = []
        c_acc_per_epoch = []

        for epoch in range(n_epochs):
            c_loss_epoch = 0.
            g_loss_epoch = 0.
            c_acc_epoch = 0.

            for X_batch in dataset:
                # perform train step on current batch
                X_batch = tf.cast(X_batch, tf.float32)
                c_loss, c_acc, g_loss = self.train_step(X_batch, batch_size)

                # save loss and accuracy per batch
                c_loss_per_batch.append(c_loss)
                c_acc_per_batch.append(c_acc)
                g_loss_per_batch.append(g_loss)
                c_loss_epoch += c_loss
                g_loss_epoch += g_loss
                c_acc_epoch += c_acc

            # aggregate epoch loss & accuracy
            c_loss_per_epoch.append(c_loss_epoch / batch_size)
            g_loss_per_epoch.append(g_loss_epoch / batch_size)
            c_acc_per_epoch.append(c_acc_epoch / batch_size)

            # logging
            print("epoch {} critic - c_loss: {} - c_accuracy: {}, generator - g_loss: {}".format(epoch, c_loss_per_epoch[-1], c_acc_per_epoch[-1], g_loss_per_epoch[-1]))

            # models checkpoint - save the model every n_checkpoint (default=10) epochs
            if (epoch + 1) % n_checkpoint == 0:
                checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

            # evaluate quality and quantitative scores every n_eval epochs

            # Add random noise to the labels - important trick! TODO

        return c_loss_per_batch, c_loss_per_epoch, g_loss_per_batch, g_loss_per_epoch, c_acc_per_batch, c_acc_per_epoch

    def generate_samples(self, samples_num: int):
        noise = tf.random.normal(shape=[samples_num, self._latent_noise_size])
        generated_samples = self.generator(noise, training=False)

        return generated_samples


class WGAN_GP(GAN):
    def __init__(self, **kwargs):
        GAN.__init__(self, kwargs)

        #self._critic_training_steps =

