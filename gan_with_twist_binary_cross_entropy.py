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
from preprocessing_utils import gather_numeric_and_categorical_columns
from utils import evaluate_using_tsne, plot_accuracy_per_epoch_history


class GANBBModelBinaryCE:
    """
    This class is implementation getting random latent noise and confidence score and based on the given black-box model
    tries to generate samples that will get similar Y output from the BB model based on the given confidence score.
    Binary-cross-entropy objective function is used.
    """
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

        self.bb_model = bb_model
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.generator_optimizer = RMSprop(learning_rate=self._generator_lr)
        self.generator_loss_fn = tf.keras.losses.BinaryCrossentropy()

    def _build_discriminator(self):
        # sample input
        sample_input = Input(shape=(self._input_size,))

        # confidence score input
        in_confidence_score = Input(shape=(1,))

        # y output of the BB model
        in_y = Input(shape=(1,))

        input_in = Concatenate()([sample_input, in_confidence_score, in_y])

        x = Dense(256, activation=self._discriminator_activation_function)(input_in)
        x = BatchNormalization()(x)
        x = Dropout(self._discriminator_dropout)(x)
        x = Dense(256, activation=self._discriminator_activation_function)(x)
        x = BatchNormalization()(x)
        x = Dropout(self._discriminator_dropout)(x)
        output = Dense(1, activation='sigmoid')(x)

        discriminator = Model([sample_input, in_confidence_score, in_y], output)

        # compile model
        opt = RMSprop(learning_rate=self._discriminator_lr)
        discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

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

    def generate_latent_points(self, n_samples):
        """generate points in latent space as input for the generator"""
        # generate points in the latent space
        z_input = tf.random.truncated_normal(shape=[self._latent_noise_size * n_samples]).numpy()
        # reshape into a batch of inputs for the network
        z_input = z_input.reshape(n_samples, self._latent_noise_size)
        # generate confidence scores
        confidence_scores = np.random.uniform(size=n_samples)

        return z_input, confidence_scores

    def generate_fake_samples(self, n_samples):
        """use the generator to generate n fake samples, with class confidence scores"""
        # generate points in latent space
        z_input, confidence_scores_input = self.generate_latent_points(n_samples)

        # predict outputs
        X = self.generator.predict([z_input, confidence_scores_input])

        return X, confidence_scores_input

    def train(self, fake_dataset_size, batch_size, gan_sample_generator, n_epochs, df_real_not_normalized, experiment_dir, logger):
        """train the generator and discriminator"""

        numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(df_real_not_normalized)

        min_mse_score_for_fixed_latent_noise_and_confidence = float('inf')
        min_mse_score_for_random_latent_noise_and_confidence = float('inf')
        samples, generated_samples, generated_confidence_scores = None, None, None
        best_epoch = 0

        d_loss_hist, g_loss_hist, d_acc_hist, score_for_fixed_latent_noise_and_confidence_hist, score_for_random_latent_noise_and_confidence_hist = list(), list(), list(), list(), list()

        # calculate the number of batches per training epoch
        batches_per_epoch = int(fake_dataset_size / batch_size)

        for epoch in range(n_epochs):
            d_loss_epoch, g_loss_epoch, d_acc_epoch = list(), list(), list()

            # enumerate batches manually
            for batch_num in range(batches_per_epoch):
                # generate 'fake' samples
                generated_samples, confidence_scores_input = self.generate_fake_samples(batch_size)

                # get BB model output for the current generated samples
                y_outputs = self.bb_model.predict_proba(generated_samples)[:, 1]

                # generate class labels
                y = randint(0, 2, batch_size)
                for i in range(batch_size):
                    if y[i] == 0:
                        temp = confidence_scores_input[i]
                        confidence_scores_input[i] = y_outputs[i]
                        y_outputs[i] = temp

                # update discriminator model weights
                d_loss, d_acc = self.discriminator.train_on_batch([generated_samples, confidence_scores_input, y_outputs], y)

                # prepare points in latent space as input for the generator
                z_input, confidence_scores = self.generate_latent_points(batch_size)

                # create inverted labels for the fake samples
                y_gan = np.zeros((batch_size, 1))

                # update the generator via the discriminator's error
                with tf.GradientTape() as tape:
                    # Generate samples using the generator
                    generated_samples = self.generator([z_input, confidence_scores], training=True)
                    # get BB model output for the current generated samples
                    y_outputs = self.bb_model.predict_proba(generated_samples)[:, 1]
                    # Get the discriminator result
                    critic_result = self.discriminator([generated_samples, confidence_scores, y_outputs], training=True)
                    # Calculate the generator loss
                    g_loss = self.generator_loss_fn(y_gan, critic_result)

                # Get the gradients w.r.t the generator loss
                gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
                # Update the weights of the generator using the generator optimizer
                self.generator_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

                g_loss_epoch.append(g_loss)
                d_loss_epoch.append(d_loss)
                d_acc_epoch.append(d_acc)

            # store loss
            g_loss_hist.append(np.mean(g_loss_epoch))
            d_loss_hist.append(np.mean(d_loss_epoch))
            d_acc_hist.append(np.mean(d_acc_epoch))

            # logging
            logger.info("epoch {} discriminator - d_loss: {} - d_acc: {}, generator - g_loss: {}".format(epoch,
                                                                                                         d_loss_hist[-1],
                                                                                                         d_acc_hist[-1],
                                                                                                         g_loss_hist[-1]))

            # # summarize performance
            samples_fixed_latent_noise, generated_samples_fixed_latent_noise, fixed_confidence_scores = gan_sample_generator.generate_samples(self.generator)
            samples_random_latent_noise, generated_samples_random_latent_noise, random_confidence_scores = gan_sample_generator.generate_samples(self.generator, random_latent_noise_and_confidence_scores=True)

            # evaluate using MSE metric between black box model outputs and confidence scores
            score_for_fixed_latent_noise_and_confidence = tf.keras.losses.MeanSquaredError()(self.bb_model.predict_proba(generated_samples_fixed_latent_noise)[:, 1], fixed_confidence_scores)
            score_for_random_latent_noise_and_confidence = tf.keras.losses.MeanSquaredError()(self.bb_model.predict_proba(generated_samples_random_latent_noise)[:, 1], random_confidence_scores)

            logger.info(
                "epoch {} MSE score for fixed latent noise and confidence scores: {}, random latent noise and random confidence scores: {}, best score(epoch={}): {}".format(
                    epoch,
                    score_for_fixed_latent_noise_and_confidence,
                    score_for_random_latent_noise_and_confidence,
                    best_epoch,
                    min_mse_score_for_fixed_latent_noise_and_confidence))

            score_for_fixed_latent_noise_and_confidence_hist.append(score_for_fixed_latent_noise_and_confidence)
            score_for_random_latent_noise_and_confidence_hist.append(score_for_random_latent_noise_and_confidence)

            if score_for_fixed_latent_noise_and_confidence < min_mse_score_for_fixed_latent_noise_and_confidence:
                min_mse_score_for_fixed_latent_noise_and_confidence = score_for_fixed_latent_noise_and_confidence
                samples = samples_fixed_latent_noise
                generated_samples = generated_samples_fixed_latent_noise
                generated_confidence_scores = fixed_confidence_scores
                best_epoch = epoch

                # save models
                self.generator.save(f"{experiment_dir}/generator.h5")
                self.discriminator.save(f"{experiment_dir}/critic.h5")

            if score_for_random_latent_noise_and_confidence < min_mse_score_for_random_latent_noise_and_confidence:
                min_mse_score_for_random_latent_noise_and_confidence = score_for_random_latent_noise_and_confidence

        # evaluate diversity using tsne
        evaluate_using_tsne(samples, np.zeros((fake_dataset_size, 1)),
                            df_real_not_normalized.columns.tolist(), categorical_columns.tolist(), best_epoch,
                            experiment_dir)
        # plot acc history
        plot_accuracy_per_epoch_history(d_acc_hist, experiment_dir)

        return d_loss_hist, g_loss_hist, min_mse_score_for_fixed_latent_noise_and_confidence,\
               min_mse_score_for_random_latent_noise_and_confidence, samples, generated_samples, generated_confidence_scores,\
               score_for_fixed_latent_noise_and_confidence_hist, score_for_random_latent_noise_and_confidence_hist
