from typing import List

import numpy as np
import tensorflow as tf
from numpy import mean
from numpy import ones
from numpy.random import randint
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate, Dropout, BatchNormalization, LeakyReLU
from tensorflow.python.keras import Input, backend, Sequential
from tensorflow.python.keras.constraints import Constraint
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from global_vars import LATENT_NOISE_SIZE, GENERATOR_LR, CRITIC_LR, CRITIC_DROPOUT, CRITIC_STEPS, SEED
from preprocessing_utils import gather_numeric_and_categorical_columns
from utils import evaluate_machine_learning_efficacy, evaluate_using_tsne


class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)


class CWGAN:
    def __init__(self,
                 input_size: int,
                 columns_size: List[int],
                 num_classes: int,
                 is_label_conditional: bool = True,
                 critic_steps: int = CRITIC_STEPS,
                 latent_noise_size: int = LATENT_NOISE_SIZE,
                 positive_negative_labels: List[int] = None,
                 **kwargs):

        tf.random.set_seed(SEED)
        np.random.seed(SEED)

        if positive_negative_labels is None:
            self.positive_negative_labels = [0, 1]
        else:
            self.positive_negative_labels = positive_negative_labels

        self._num_classes = num_classes
        self._columns_size = columns_size
        self._input_size = input_size
        self._critic_steps = critic_steps
        self._latent_noise_size = latent_noise_size
        self._is_label_conditional = is_label_conditional

        self._generator_activation_function = kwargs.get('generator_activation_function', LeakyReLU(alpha=0.2))
        self._critic_activation_function = kwargs.get('critic_activation_function', LeakyReLU(alpha=0.2))
        self._generator_lr = kwargs.get('generator_lr', GENERATOR_LR)
        self._critic_lr = kwargs.get('critic_lr', CRITIC_LR)
        self._critic_dropout = kwargs.get('critic_dropout', CRITIC_DROPOUT)

        self.generator = self._build_generator()
        self.critic = self._build_critic()
        self.gan = self._define_gan(self.generator, self.critic)

    def _build_critic(self):
        # sample input
        sample_input = Input(shape=(self._input_size,))

        if self._is_label_conditional:
            # label input
            in_label = Input(shape=(1,))

            input_in = Concatenate()([sample_input, in_label])
        else:
            input_in = sample_input

        weights_constraint = ClipConstraint(0.01)

        x = Dense(256, activation=self._critic_activation_function, kernel_constraint=weights_constraint)(input_in)
        x = BatchNormalization()(x)
        x = Dropout(self._critic_dropout)(x)
        x = Dense(256, activation=self._critic_activation_function, kernel_constraint=weights_constraint)(x)
        x = BatchNormalization()(x)
        x = Dropout(self._critic_dropout)(x)
        output = Dense(1)(x)

        if self._is_label_conditional:
            critic = Model([sample_input, in_label], output)
        else:
            critic = Model(sample_input, output)

        # compile model
        opt = RMSprop(learning_rate=self._critic_lr)
        critic.compile(loss=wasserstein_loss, optimizer=opt)

        return critic

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

    def _define_gan(self, generator, critic):
        # make weights in the critic not trainable
        critic.trainable = False

        if self._is_label_conditional:
            # get noise and label inputs from generator model
            gen_noise, gen_label = generator.input

            # get output from the generator model
            gen_output = generator.output

            # connect generator output and label input from generator as inputs to critic
            gan_output = critic([gen_output, gen_label])

            # define gan model as taking noise and label and outputting a classification
            model = Model([gen_noise, gen_label], gan_output)
        else:
            model = Sequential()
            # add generator
            model.add(generator)
            # add the critir
            model.add(critic)

        opt = RMSprop(learning_rate=self._generator_lr)
        model.compile(loss=wasserstein_loss, optimizer=opt)

        return model

    def generate_real_samples(self, X, y_labels, n_samples):
        # choose random instances
        ix = randint(0, X.shape[0], n_samples)
        # select samples
        X = X[ix]
        labels = y_labels[ix]

        # generate class labels, -1 for 'real'
        y = -ones((n_samples, 1))

        return X, labels, y

    def generate_latent_points(self, n_samples):
        """generate points in latent space as input for the generator"""
        # generate points in the latent space
        z_input = tf.random.truncated_normal(shape=[self._latent_noise_size * n_samples]).numpy()
        # reshape into a batch of inputs for the network
        z_input = z_input.reshape(n_samples, self._latent_noise_size)
        # generate labels
        labels = np.random.choice(self.positive_negative_labels, n_samples)

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

        # create class labels with 1.0 for 'fake'
        y = ones((n_samples, 1))

        return X, labels_input, y

    def train(self, X, y, batch_size, gan_sample_generator, X_test, y_test, n_epochs, df_real_not_normalized, experiment_dir, logger):
        """train the generator and critic"""

        numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(df_real_not_normalized)

        # calculate the number of batches per training epoch
        batches_per_epoch = int(X.shape[0] / batch_size)
        # calculate the number of training iterations
        n_steps = batches_per_epoch * n_epochs
        # calculate the size of half a batch of samples
        half_batch = int(batch_size / 2)
        # lists for keeping track of loss
        c1_epoch_hist, c2_epoch_hist, g_epoch_hist, c1_hist, c2_hist, g_hist, score_for_fixed_latent_noise_hist, score_for_random_latent_noise_hist = list(), list(), list(), list(), list(), list(), list(), list()

        max_score_for_fixed_latent_noise = 0.
        max_score_for_random_latent_noise = 0.
        samples, generated_samples, generated_labels = None, None, None
        best_epoch = 0

        # manually enumerate epochs
        epoch = 1
        for i in range(n_steps):
            # update the critic more than the generator
            c1_tmp, c2_tmp = list(), list()
            for _ in range(self._critic_steps):
                # get randomly selected 'real' samples
                X_real, labels_input, y_real = self.generate_real_samples(X, y, half_batch)
                # update critic model weights
                if self._is_label_conditional:
                    c_loss1 = self.critic.train_on_batch([X_real, labels_input], y_real)
                else:
                    c_loss1 = self.critic.train_on_batch(X_real, y_real)
                c1_tmp.append(c_loss1)
                # generate 'fake' examples
                X_fake, labels_input, y_fake = self.generate_fake_samples(half_batch)
                # update critic model weights
                if self._is_label_conditional:
                    c_loss2 = self.critic.train_on_batch([X_fake, labels_input], y_fake)
                else:
                    c_loss2 = self.critic.train_on_batch(X_fake, y_fake)
                c2_tmp.append(c_loss2)
            # store critic loss
            c1_hist.append(mean(c1_tmp))
            c2_hist.append(mean(c2_tmp))
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = self.generate_latent_points(batch_size)
            # create inverted labels for the fake samples
            y_gan = -ones((batch_size, 1))

            # update the generator via the critic's error
            if self._is_label_conditional:
                g_loss = self.gan.train_on_batch([z_input, labels_input], y_gan)
            else:
                g_loss = self.gan.train_on_batch(z_input, y_gan)
            g_hist.append(g_loss)

            # evaluate the model performance every 'epoch'
            if (i + 1) % batches_per_epoch == 0:
                c1_epoch_hist.append(mean(c1_hist))
                c2_epoch_hist.append(mean(c2_hist))
                g_epoch_hist.append(mean(g_hist))

                # summarize loss on this batch
                logger.info("epoch {} critic - c_loss1: {} - c_loss2: {}, generator - g_loss: {}".format(epoch,
                                                                                                         c1_epoch_hist[-1],
                                                                                                         c2_epoch_hist[-1],
                                                                                                         g_epoch_hist[-1]))
                c1_hist = list()
                c2_hist = list()
                g_hist = list()

                # summarize performance
                samples_fixed_latent_noise, generated_samples_fixed_latent_noise, labels_fixed_latent_noise = gan_sample_generator.generate_samples(self.generator)
                samples_random_latent_noise, generated_samples_random_latent_noisee, labels_random_latent_noise = gan_sample_generator.generate_samples(self.generator, random_latent_noise=True)

                # evaluate using machine learning efficacy
                score_for_fixed_latent_noise = evaluate_machine_learning_efficacy(generated_samples_fixed_latent_noise, labels_fixed_latent_noise, X_test, y_test)
                score_for_random_latent_noise = evaluate_machine_learning_efficacy(generated_samples_random_latent_noisee, labels_random_latent_noise, X_test, y_test)

                logger.info("epoch {} ML efficacy score fixed latent noise: {}, random latent noise: {}, best score(epoch={}): {}".format(
                        epoch,
                        score_for_fixed_latent_noise,
                        score_for_random_latent_noise,
                        best_epoch,
                        max_score_for_fixed_latent_noise))

                score_for_fixed_latent_noise_hist.append(score_for_fixed_latent_noise)
                score_for_random_latent_noise_hist.append(score_for_random_latent_noise)

                if score_for_fixed_latent_noise > max_score_for_fixed_latent_noise:
                    max_score_for_fixed_latent_noise = score_for_fixed_latent_noise
                    samples = samples_fixed_latent_noise
                    generated_samples = generated_samples_fixed_latent_noise
                    generated_labels = labels_fixed_latent_noise
                    best_epoch = epoch

                    # save models
                    self.generator.save(f"{experiment_dir}/generator.h5")
                    self.critic.save(f"{experiment_dir}/critic.h5")
                    self.gan.save(f"{experiment_dir}/gan.h5")

                if score_for_random_latent_noise > max_score_for_random_latent_noise:
                    max_score_for_random_latent_noise = score_for_random_latent_noise

                epoch += 1

        # evaluate using tsne
        evaluate_using_tsne(samples, generated_labels,
                            df_real_not_normalized.columns.tolist(), categorical_columns.tolist(), best_epoch,
                            experiment_dir)

        return c1_epoch_hist, c2_epoch_hist, g_epoch_hist, max_score_for_fixed_latent_noise, max_score_for_random_latent_noise, samples,\
               generated_samples, generated_labels, score_for_fixed_latent_noise_hist, score_for_random_latent_noise_hist
