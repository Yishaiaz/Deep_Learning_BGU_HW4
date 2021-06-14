from typing import List

import numpy as np
import tensorflow as tf
from numpy.random import randint
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate, Dropout, BatchNormalization, LeakyReLU
from tensorflow.python.keras import Input
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from global_vars import LATENT_NOISE_SIZE, GENERATOR_LR, CRITIC_LR, CRITIC_DROPOUT, SEED


class GANBBModel:
    def __init__(self,
                 bb_model: RandomForestClassifier,
                 input_size: int,
                 columns_size: List[int],
                 latent_noise_size: int = LATENT_NOISE_SIZE,
                 **kwargs):

        tf.random.set_seed(SEED)
        np.random.seed(SEED)

        self._columns_size = columns_size
        self._input_size = input_size
        self._latent_noise_size = latent_noise_size

        self._generator_activation_function = kwargs.get('generator_activation_function', LeakyReLU(alpha=0.2))
        self._discriminator_activation_function = kwargs.get('discriminator_activation_function', LeakyReLU(alpha=0.2))
        self._generator_lr = kwargs.get('generator_lr', GENERATOR_LR)
        self._discriminator_lr = kwargs.get('discriminator_lr', CRITIC_LR)
        self._discriminator_dropout = kwargs.get('discriminator_dropout', CRITIC_DROPOUT)

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._define_gan(self.generator, self.discriminator)
        self.bb_model = bb_model

    def _build_discriminator(self):
        # sample input
        sample_input = Input(shape=(self._input_size,))

        # confidence score input
        in_confidence_score = Input(shape=(1,))

        # y output of the BB model
        in_y = Input(shape=(1,))

        input_in = Concatenate()([sample_input, in_confidence_score, in_y])

        x = Dense(128, activation=self._discriminator_activation_function)(input_in)
        x = BatchNormalization()(x)
        x = Dropout(self._discriminator_dropout)(x)
        x = Dense(128, activation=self._discriminator_activation_function)(x)
        x = BatchNormalization()(x)
        x = Dropout(self._discriminator_dropout)(x)
        output = Dense(1)(x)

        discriminator = Model([sample_input, in_confidence_score, in_y], output)

        # compile model
        opt = RMSprop(learning_rate=self._discriminator_lr)
        discriminator.compile(loss='mse', optimizer=opt)

        return discriminator

    def _build_generator(self):
        # latent noise input
        noise_input = Input(shape=(self._latent_noise_size,))

        # confidence score input
        in_confidence_score = Input(shape=(1,))

        input_in = Concatenate()([noise_input, in_confidence_score])

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

        generator = Model(inputs=[noise_input, in_confidence_score], outputs=output)

        return generator

    def _define_gan(self, generator, discriminator, bb_model):
        # make weights in the discriminator not trainable
        discriminator.trainable = False

        # get noise and confidence_score inputs from generator model
        gen_noise, gen_confidence_score = generator.input

        # get output from the generator model
        gen_output = generator.output

        # get y output of the BB model
        bb_model_output = bb_model(gen_output)

        # connect generator output and confidence_score input from generator and BB model y output as inputs to discriminator
        gan_output = discriminator([gen_output, gen_confidence_score, bb_model_output])

        # define gan model as taking noise and confidence_score and outputting a classification
        model = Model([gen_noise, gen_confidence_score], gan_output)

        opt = RMSprop(learning_rate=self._generator_lr)
        model.compile(loss='mse', optimizer=opt)

        return model

    def generate_latent_points(self, n_samples):
        """generate points in latent space as input for the generator"""
        # generate points in the latent space
        z_input = tf.random.truncated_normal(shape=[self._latent_noise_size * n_samples]).numpy()
        # reshape into a batch of inputs for the network
        z_input = z_input.reshape(n_samples, self._latent_noise_size)
        # generate confidence scores
        confidence_scores = np.random.uniform(n_samples)

        return z_input, confidence_scores

    def generate_fake_samples(self, n_samples):
        """use the generator to generate n fake samples, with class confidence scores"""
        # generate points in latent space
        z_input, confidence_scores_input = self.generate_latent_points(n_samples)

        # predict outputs
        X = self.generator.predict([z_input, confidence_scores_input])

        return X, confidence_scores_input

    def train(self, fake_dataset_size, batch_size, gan_sample_generator, n_epochs, experiment_dir, logger):
        """train the generator and discriminator"""

        max_score_for_fixed_latent_noise = 0.
        max_score_for_random_latent_noise = 0.
        samples, generated_samples, generated_labels = None, None, None

        d_loss_hist, g_loss_hist = list(), list()

        # calculate the number of batches per training epoch
        batches_per_epoch = int(fake_dataset_size / batch_size)

        for epoch in range(n_epochs):
            d_loss_epoch, g_loss_epoch = list(), list()

            # enumerate batches manually
            for batch_num in range(batches_per_epoch):
                # generate 'fake' samples
                X_fake, confidence_scores_input = self.generate_fake_samples(batch_size)

                # get BB model output for the current generated samples
                y_outputs = self.bb_model.predict(X_fake)

                # generate class labels
                y = randint(0, 2, batch_size)
                for i in range(batch_size):
                    if y == 0:
                        temp = confidence_scores_input[i]
                        confidence_scores_input[i] = y_outputs[i]
                        y_outputs[i] = temp

                # update discriminator model weights
                d_loss = self.discriminator.train_on_batch([X_fake, confidence_scores_input, y_outputs], y)

                # prepare points in latent space as input for the generator
                z_input, confidence_scores = self.generate_latent_points(batch_size)

                # create inverted labels for the fake samples
                y_gan = np.zeros((batch_size, 1))

                # update the generator via the discriminator's error
                g_loss = self.gan.train_on_batch([z_input, confidence_scores_input], y_gan)

                g_loss_epoch.append(g_loss)
                d_loss_epoch.append(d_loss)

            # store loss & accuracy
            g_loss_hist.append(np.mean(g_loss_epoch))
            d_loss_hist.append(np.mean(d_loss_epoch))

            # logging
            logger.info("epoch {} discriminator - d_loss: {}, generator - g_loss: {}".format(epoch,
                                                                                             d_loss_hist[-1],
                                                                                             g_loss_hist[-1]))

            # # summarize performance
            # samples_fixed_latent_noise, generated_samples_fixed_latent_noise, labels_fixed_latent_noise = gan_sample_generator.generate_samples(self.generator)
            # samples_random_latent_noise, generated_samples_random_latent_noise, labels_random_latent_noise = gan_sample_generator.generate_samples(self.generator, random_latent_noise=True)
            #
            # # evaluate using machine learning efficacy
            # score_for_fixed_latent_noise = evaluate_machine_learning_efficacy(generated_samples_fixed_latent_noise, labels_fixed_latent_noise, X_test, y_test)
            # score_for_random_latent_noise = evaluate_machine_learning_efficacy(generated_samples_random_latent_noise, labels_random_latent_noise, X_test, y_test)
            #
            # logger.info("epoch {} ML efficacy score fixed latent noise: {}, random latent noise: {}".format(epoch,
            #                                                                                                 score_for_fixed_latent_noise,
            #                                                                                                 score_for_random_latent_noise))
            #
            # if score_for_fixed_latent_noise > max_score_for_fixed_latent_noise:
            #     max_score_for_fixed_latent_noise = score_for_fixed_latent_noise
            #     samples = samples_fixed_latent_noise
            #     generated_samples = generated_samples_fixed_latent_noise
            #     generated_labels = labels_fixed_latent_noise
            #
            #     # save models
            #     self.generator.save(f"{experiment_dir}/generator.h5")
            #     self.discriminator.save(f"{experiment_dir}/critic.h5")
            #     self.gan.save(f"{experiment_dir}/gan.h5")
            #
            # if score_for_random_latent_noise > max_score_for_random_latent_noise:
            #     max_score_for_random_latent_noise = score_for_random_latent_noise

        return d_loss_hist, g_loss_hist, max_score_for_fixed_latent_noise, max_score_for_random_latent_noise, samples, generated_samples, generated_labels
