from typing import List

import numpy as np
import tensorflow as tf
from numpy.random import randint
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate, Dropout, BatchNormalization, LeakyReLU
from tensorflow.python.data import Dataset
from tensorflow.python.keras import Input
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from global_vars import LATENT_NOISE_SIZE, GENERATOR_LR, CRITIC_LR, CHECKPOINT_PREFIX, SEED, \
    CRITIC_DROPOUT


class GAN:
    def __init__(self,
                 input_size: int,
                 columns_size: List[int],
                 num_classes: int,
                 **kwargs):

        tf.random.set_seed(SEED)
        np.random.seed(SEED)

        self._input_size = input_size
        self._columns_size = columns_size
        self._num_classes = num_classes

        self._latent_noise_size = kwargs.get('latent_size', LATENT_NOISE_SIZE)
        self._generator_activation_function = kwargs.get('generator_activation_function', 'relu')
        self._critic_activation_function = kwargs.get('critic_activation_function', LeakyReLU())
        self._generator_lr = kwargs.get('generator_lr', GENERATOR_LR)
        self._critic_lr = kwargs.get('critic_lr', CRITIC_LR)
        self._critic_dropout = kwargs.get('critic_dropout', CRITIC_DROPOUT)

        self.critic = self._build_critic()
        self._compile_critic()

        self.generator = self._build_generator()
        self.gan = self._build_gan()
        self._compile_gan()

    def _build_gan(self):
        # make weights in the critic not trainable
        self.critic.trainable = False

        # get noise and label inputs from generator model
        gen_noise, gen_label = self.generator.input

        # get output from the generator model
        gen_output = self.generator.output

        # connect generator output and label input from generator as inputs to critic
        gan_output = self.critic([gen_output, gen_label])

        # define gan model as taking noise and label and outputting a classification
        model = Model([gen_noise, gen_label], gan_output)

        return model

    def _build_generator(self):
        # label input
        in_label = Input(shape=(1,))

        # latent noise input
        noise_input = Input(shape=(self._latent_noise_size,))

        concat_input = Concatenate()([noise_input, in_label])

        x = Dense(256, activation=self._generator_activation_function, kernel_initializer='he_uniform')(concat_input)
        x = BatchNormalization()(x)
        x = Dense(256, activation=self._generator_activation_function, kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)

        layers = []
        for column_size in self._columns_size:
            if column_size == 1:
                layers.append(Dense(1, activation='tanh')(x))
            else:
                layers.append(Dense(column_size, activation='softmax')(x))

        output = Concatenate()(layers)

        generator = Model(inputs=[noise_input, in_label], outputs=output)

        return generator

    def _build_critic(self):
        # label input
        in_label = Input(shape=(1,))

        # sample input
        in_input = Input(shape=(self._input_size,))

        concat_input = Concatenate()([in_input, in_label])

        x = Dense(256, activation=self._critic_activation_function, kernel_initializer='he_uniform')(concat_input)
        x = BatchNormalization()(x)
        x = Dropout(self._critic_dropout)(x)
        x = Dense(256, activation=self._critic_activation_function, kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Dropout(self._critic_dropout)(x)
        output = Dense(1, activation='sigmoid')(x)

        critic = Model([in_input, in_label], output)

        return critic

    def _compile_critic(self):
        self._critic_optimizer = RMSprop(learning_rate=self._critic_lr)
        self.critic.compile(loss="binary_crossentropy", optimizer=self._critic_optimizer, metrics=['accuracy'])

    def _compile_gan(self):
        self._generator_optimizer = RMSprop(learning_rate=self._generator_lr)
        self.gan.compile(loss="binary_crossentropy", optimizer=self._generator_optimizer)

    def _generate_labels(self, batch_size: int):
        labels = tf.convert_to_tensor(randint(0, self._num_classes, batch_size))
        labels = tf.cast(labels, tf.int64)
        labels = tf.reshape(labels, [batch_size, 1])

        return labels

    def train_step(self, X_batch, y_batch, batch_size: int):
        generator = self.generator
        critic = self.critic

        # phase 1 - training the critic
        noise = tf.random.normal(shape=[batch_size, self._latent_noise_size])
        labels = self._generate_labels(batch_size)

        generated_samples = generator([noise, labels])

        X_fake_and_real = tf.concat([generated_samples, X_batch], axis=0)
        y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)

        # generate labels
        labels = tf.concat([labels, y_batch], 0)

        critic.trainable = True
        c_loss, c_acc = critic.train_on_batch([X_fake_and_real, labels], y1)

        # phase 2 - training the generator
        noise = tf.random.normal(shape=[batch_size, self._latent_noise_size])
        # create inverted labels for the fake samples
        y2 = tf.constant([[1.]] * batch_size)

        # generate labels
        labels = self._generate_labels(batch_size)

        critic.trainable = False
        g_loss = self.gan.train_on_batch([noise, labels], y2)

        return c_loss, c_acc, g_loss

    def train_gan(self,
                  dataset: Dataset,
                  batch_size: int,
                  n_epochs: int,
                  n_eval: int = 15,
                  n_checkpoint: int = 10):

        generator = self.generator
        critic = self.critic

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

            for X_batch, y_batch in dataset:
                # perform train step on current batch
                X_batch = tf.cast(X_batch, tf.float32)

                c_loss, c_acc, g_loss = self.train_step(X_batch, y_batch, batch_size)

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


            # models checkpoint - save the model every n_checkpoint (default=10) epochs
            if (epoch + 1) % n_checkpoint == 0:
                checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

            # evaluate quality and quantitative scores every n_eval epochs

            # Add random noise to the labels - important trick! TODO

        return c_loss_per_batch, c_loss_per_epoch, g_loss_per_batch, g_loss_per_epoch, c_acc_per_batch, c_acc_per_epoch

    def generate_samples(self,
                         samples_num: int,
                         label: int,
                         columns_size: List[int]):
        noise = tf.random.normal(shape=[samples_num, self._latent_noise_size])
        generated_samples = self.generator([noise, [label]*samples_num], training=False)

        # for generated_sample in generated_samples:
        #     for column_size in columns_size:
        #         if columns_size > 1:
        #

        return generated_samples

#
# class WGAN_GP(GAN):
#     def __init__(self,
#                  input_size: int,
#                  columns_size: List[int],
#                  num_classes: int,
#                  **kwargs):
#         GAN.__init__(self, input_size, columns_size, num_classes, **kwargs)
#
#         self._critic_steps = kwargs.get('critic_steps', CRITIC_STEPS)
#         self._gp_weight = kwargs.get('gp_weight', GP_WEIGHT)
#
#     @tf.function
#     def gradient_penalty(self, batch_size, X_real, X_fake, labels):
#         """
#         Calculates the gradient penalty.
#         This loss is calculated on an interpolated data and added to the critic loss.
#         """
#         # Get the interpolated samples
#         alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
#         diff = X_fake - X_real
#         interpolated = X_real + alpha * diff
#
#         with tf.GradientTape() as gp_tape:
#             gp_tape.watch(interpolated)
#             gp_tape.watch(labels)
#
#             # 1. Get the critic output for this interpolated samples.
#             pred = self.critic([interpolated, labels], training=True)
#
#         # 2. Calculate the gradients w.r.t to this interpolated samples.
#         grads = gp_tape.gradient(pred, [interpolated])[0] # TODO
#         # 3. Calculate the norm of the gradients.
#         norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1])) # TODO
#         gp = tf.reduce_mean((norm - 1.0) ** 2)
#
#         return gp
#
#     @staticmethod
#     def critic_loss(real_samples, fake_samples):
#         real_loss = tf.reduce_mean(real_samples)
#         fake_loss = tf.reduce_mean(fake_samples)
#         return fake_loss - real_loss
#
#     @staticmethod
#     def generator_loss(fake_samples):
#         return -tf.reduce_mean(fake_samples)
#
#     @tf.function
#     def train_step(self, X_batch, y_batch, batch_size: int):
#         # Note: this implementation is based on https://keras.io/examples/generative/wgan_gp/ with minor changes
#         # For each batch, we are going to perform the
#         # following steps as laid out in the original paper:
#         # 1. Train the generator and get the generator loss
#         # 2. Train the critic and get the critic loss
#         # 3. Calculate the gradient penalty
#         # 4. Multiply this gradient penalty with a constant weight factor
#         # 5. Add the gradient penalty to the critic loss
#         # 6. Return the generator and critic losses as a loss tuple plus the critic's accuracy
#
#         # Train the critic first. The original paper recommends training
#         # the critic for `x` more steps (typically 5) as compared to
#         # one step of the generator.
#         for i in range(self._critic_steps):
#             # Get the latent vector
#             random_latent_vectors = tf.random.normal(shape=(batch_size, self._latent_noise_size))
#
#             # generate labels
#             labels = self._generate_labels(batch_size)
#
#             with tf.GradientTape() as tape:
#                 # Generate fake samples from the latent vector
#                 generated_samples = self.generator([random_latent_vectors, labels], training=True)
#                 # Get the logits for the fake samples
#                 fake_logits = self.critic([generated_samples, labels], training=True)
#                 # Get the logits for the real samples
#                 real_logits = self.critic([X_batch, y_batch], training=True)
#
#                 # Calculate the critic loss using the fake and real samples logits
#                 c_cost = WGAN_GP.critic_loss(real_logits, fake_logits)
#                 # Calculate the gradient penalty
#                 #gp = self.gradient_penalty(batch_size, X_batch, generated_samples, y_batch)
#                 # Add the gradient penalty to the original critic loss
#                 #critic_loss = c_cost + gp * self._gp_weight
#                 critic_loss = c_cost
#
#             # Get the gradients w.r.t the critic loss
#             d_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
#             # Update the weights of the critic using the critic optimizer
#             self._critic_optimizer.apply_gradients(zip(d_gradient, self.critic.trainable_variables))
#
#         # Train the generator
#         # Get the latent vector
#         random_latent_vectors = tf.random.normal(shape=(batch_size, self._latent_noise_size))
#         # generate labels
#         labels = self._generate_labels(batch_size)
#
#         with tf.GradientTape() as tape:
#             # Generate fake samples using the generator
#             generated_samples = self.generator([random_latent_vectors, labels], training=True)
#             # Get the critic logits for fake samples
#             gen_samples_logits = self.critic([generated_samples, labels], training=True)
#             # Calculate the generator loss
#             g_loss = WGAN_GP.generator_loss(gen_samples_logits)
#
#         # Get the gradients w.r.t the generator loss
#         gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
#         # Update the weights of the generator using the generator optimizer
#         self._generator_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
#
#         return critic_loss, 0, g_loss  # TODO