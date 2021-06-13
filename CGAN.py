from typing import List

import numpy as np
import tensorflow as tf
from numpy import ones
from numpy.random import randint
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate, Dropout, BatchNormalization, LeakyReLU
from tensorflow.python.keras import Input, Sequential
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from global_vars import LATENT_NOISE_SIZE, GENERATOR_LR, CRITIC_LR, CRITIC_DROPOUT, SEED
from utils import evaluate_machine_learning_efficacy, evaluate_using_tsne


class CGAN:
    def __init__(self,
                 input_size: int,
                 columns_size: List[int],
                 num_classes: int,
                 is_label_conditional: bool = True,
                 latent_noise_size: int = LATENT_NOISE_SIZE,
                 **kwargs):

        tf.random.set_seed(SEED)
        np.random.seed(SEED)

        self._num_classes = num_classes
        self._columns_size = columns_size
        self._input_size = input_size
        self._latent_noise_size = latent_noise_size
        self._is_label_conditional = is_label_conditional

        self._generator_activation_function = kwargs.get('generator_activation_function', LeakyReLU(alpha=0.2))
        self._discriminator_activation_function = kwargs.get('discriminator_activation_function', LeakyReLU(alpha=0.2))
        self._generator_lr = kwargs.get('generator_lr', GENERATOR_LR)
        self._discriminator_lr = kwargs.get('discriminator_lr', CRITIC_LR)
        self._discriminator_dropout = kwargs.get('discriminator_dropout', CRITIC_DROPOUT)

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._define_gan(self.generator, self.discriminator)

    def _build_discriminator(self):
        # sample input
        sample_input = Input(shape=(self._input_size,))

        if self._is_label_conditional:
            # label input
            in_label = Input(shape=(1,))

            input_in = Concatenate()([sample_input, in_label])
        else:
            input_in = sample_input

        x = Dense(128, activation=self._discriminator_activation_function)(input_in)
        x = BatchNormalization()(x)
        x = Dropout(self._discriminator_dropout)(x)
        x = Dense(128, activation=self._discriminator_activation_function)(x)
        x = BatchNormalization()(x)
        x = Dropout(self._discriminator_dropout)(x)
        output = Dense(1, activation='sigmoid')(x)

        if self._is_label_conditional:
            discriminator = Model([sample_input, in_label], output)
        else:
            discriminator = Model(sample_input, output)

        # compile model
        opt = RMSprop(learning_rate=self._discriminator_lr)
        discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        return discriminator

    def _build_generator(self):
        # latent noise input
        noise_input = Input(shape=(self._latent_noise_size,))

        if self._is_label_conditional:
            # label input
            in_label = Input(shape=(1,))

            input_in = Concatenate()([noise_input, in_label])
        else:
            input_in = noise_input

        x = Dense(256, activation=self._generator_activation_function)(input_in)
        x = BatchNormalization()(x)
        x = Dense(256, activation=self._generator_activation_function)(x)
        x = BatchNormalization()(x)

        layers = []
        for column_size in self._columns_size:
            if column_size == 1:
                layers.append(Dense(1, activation='tanh')(x))
            else:
                layers.append(Dense(column_size, activation='softmax')(x))

        output = Concatenate()(layers)

        if self._is_label_conditional:
            generator = Model(inputs=[noise_input, in_label], outputs=output)
        else:
            generator = Model(inputs=noise_input, outputs=output)

        return generator

    def _define_gan(self, generator, discriminator):
        # make weights in the discriminator not trainable
        discriminator.trainable = False

        if self._is_label_conditional:
            # get noise and label inputs from generator model
            gen_noise, gen_label = generator.input

            # get output from the generator model
            gen_output = generator.output

            # connect generator output and label input from generator as inputs to discriminator
            gan_output = discriminator([gen_output, gen_label])

            # define gan model as taking noise and label and outputting a classification
            model = Model([gen_noise, gen_label], gan_output)
        else:
            model = Sequential()
            # add generator
            model.add(generator)
            # add the discriminator
            model.add(discriminator)

        opt = RMSprop(learning_rate=self._generator_lr)
        model.compile(loss='binary_crossentropy', optimizer=opt)

        return model

    def generate_latent_points(self, n_samples):
        """generate points in latent space as input for the generator"""
        # generate points in the latent space
        z_input = tf.random.truncated_normal(shape=[self._latent_noise_size * n_samples]).numpy()
        # reshape into a batch of inputs for the network
        z_input = z_input.reshape(n_samples, self._latent_noise_size)
        # generate labels
        labels = randint(0, self._num_classes, n_samples)

        return z_input, labels

    def generate_fake_samples(self, n_samples):
        """use the generator to generate n fake examples, with class labels"""
        # generate points in latent space
        z_input, labels_input = self.generate_latent_points(n_samples)

        # predict outputs
        if self._is_label_conditional:
            X = self.generator.predict([z_input, labels_input])
        else:
            X = self.generator.predict(z_input)

        # create class labels
        y = np.zeros((n_samples, 1))

        return X, labels_input, y

    def train(self, dataset, batch_size, gan_sample_generator, X_test, y_test, n_epochs, df_columns, experiment_dir, logger):
        """train the generator and discriminator"""

        max_score_for_fixed_latent_noise = 0.
        max_score_for_random_latent_noise = 0.
        samples, generated_samples, generated_labels = None, None, None

        d_loss1_hist, d_loss2_hist, g_loss_hist, d_acc1_hist, d_acc2_hist = list(), list(), list(), list(), list()

        for epoch in range(n_epochs):
            d_loss1_epoch, d_loss2_epoch, g_loss_epoch, d_acc1_epoch, d_acc2_epoch = list(), list(), list(), list(), list()

            # enumerate batches over the training set
            for X_batch, y_batch in dataset:
                # generate class labels
                y_real = ones((batch_size, 1))

                # update discriminator model weights
                if self._is_label_conditional:
                    d_loss1, d_acc1 = self.discriminator.train_on_batch([X_batch, y_batch], y_real)
                else:
                    d_loss1, d_acc1 = self.discriminator.train_on_batch(X_batch, y_real)

                # generate 'fake' examples
                X_fake, labels, y_fake = self.generate_fake_samples(batch_size)

                # update discriminator model weights
                if self._is_label_conditional:
                    d_loss2, d_acc2 = self.discriminator.train_on_batch([X_fake, labels], y_fake)
                else:
                    d_loss2, d_acc2 = self.discriminator.train_on_batch(X_fake, y_fake)

                # prepare points in latent space as input for the generator
                [z_input, labels_input] = self.generate_latent_points(batch_size)
                # create inverted labels for the fake samples
                y_gan = ones((batch_size, 1))

                # update the generator via the discriminator's error
                if self._is_label_conditional:
                    g_loss = self.gan.train_on_batch([z_input, labels_input], y_gan)
                else:
                    g_loss = self.gan.train_on_batch(z_input, y_gan)

                g_loss_epoch.append(g_loss)
                d_loss1_epoch.append(d_loss1)
                d_loss2_epoch.append(d_loss2)
                d_acc1_epoch.append(d_acc1)
                d_acc2_epoch.append(d_acc2)

            # store loss & accuracy
            g_loss_hist.append(np.mean(g_loss_epoch))
            d_loss1_hist.append(np.mean(d_loss1_epoch))
            d_loss2_hist.append(np.mean(d_loss2_epoch))
            d_acc1_hist.append(np.mean(d_acc1_epoch))
            d_acc2_hist.append(np.mean(d_acc2_epoch))

            # logging
            logger.info("epoch {} discriminator - d_loss1: {} - d_loss2: {} - d_acc1: {} - d_acc2: {}, generator - g_loss: {}"
                  .format(epoch,
                          d_loss1_hist[-1],
                          d_loss2_hist[-1],
                          d_acc1_hist[-1],
                          d_acc2_hist[-1],
                          g_loss_hist[-1]))

            # summarize performance
            samples_fixed_latent_noise, generated_samples_fixed_latent_noise, labels_fixed_latent_noise = gan_sample_generator.generate_samples(self.generator)
            samples_random_latent_noise, generated_samples_random_latent_noise, labels_random_latent_noise = gan_sample_generator.generate_samples(self.generator, random_latent_noise=True)

            # evaluate using machine learning efficacy
            score_for_fixed_latent_noise = evaluate_machine_learning_efficacy(generated_samples_fixed_latent_noise, labels_fixed_latent_noise, X_test, y_test)
            score_for_random_latent_noise = evaluate_machine_learning_efficacy(generated_samples_random_latent_noise, labels_random_latent_noise, X_test, y_test)

            logger.info("epoch {} ML efficacy score fixed latent noise: {}, random latent noise: {}".format(epoch,
                                                                                                            score_for_fixed_latent_noise,
                                                                                                            score_for_random_latent_noise))

            if score_for_fixed_latent_noise > max_score_for_fixed_latent_noise:
                max_score_for_fixed_latent_noise = score_for_fixed_latent_noise
                samples = samples_fixed_latent_noise
                generated_samples = generated_samples_fixed_latent_noise
                generated_labels = labels_fixed_latent_noise

                # save models
                self.generator.save(f"{experiment_dir}/generator.h5")
                self.discriminator.save(f"{experiment_dir}/critic.h5")
                self.gan.save(f"{experiment_dir}/gan.h5")

                # evaluate using tsne
                evaluate_using_tsne(samples_fixed_latent_noise, generated_labels, df_columns, epoch, experiment_dir)

            if score_for_random_latent_noise > max_score_for_random_latent_noise:
                max_score_for_random_latent_noise = score_for_random_latent_noise

        return d_loss1_hist, d_loss2_hist, g_loss_hist, d_acc1_hist,  d_acc2_hist,\
               max_score_for_fixed_latent_noise, max_score_for_random_latent_noise, samples, generated_samples, generated_labels
