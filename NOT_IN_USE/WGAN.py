from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import Callback
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate, Dropout, BatchNormalization, LeakyReLU
from tensorflow.python.keras import Input, backend
from tensorflow.python.keras.constraints import Constraint
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from global_vars import LATENT_NOISE_SIZE, GENERATOR_LR, CRITIC_LR, CRITIC_DROPOUT, CRITIC_STEPS, GP_WEIGHT, \
    CHECKPOINT_PREFIX, SEED
from preprocessing_utils import gather_numeric_and_categorical_columns
from utils import tsne


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


class WGAN(Model):
    def __init__(self,
                 input_size: int,
                 columns_size: List[int],
                 num_classes: int,
                 critic_steps: int = CRITIC_STEPS,
                 gp_weight: float = GP_WEIGHT,
                 latent_noise_size: int = LATENT_NOISE_SIZE,
                 **kwargs):

        super(WGAN, self).__init__()

        self._num_classes = num_classes
        self._columns_size = columns_size
        self._input_size = input_size
        self._critic_steps = critic_steps
        self._gp_weight = gp_weight
        self._latent_noise_size = latent_noise_size

        self._generator_activation_function = kwargs.get('generator_activation_function', 'relu')
        self._critic_activation_function = kwargs.get('critic_activation_function', LeakyReLU())
        self._generator_lr = kwargs.get('generator_lr', GENERATOR_LR)
        self._critic_lr = kwargs.get('critic_lr', CRITIC_LR)
        self._critic_dropout = kwargs.get('critic_dropout', CRITIC_DROPOUT)

        self.generator = self._build_generator()
        self.critic = self._build_critic()

        self.generator_optimizer = RMSprop(learning_rate=0.00005)#Adam(learning_rate=GENERATOR_LR, beta_1=0.5, beta_2=0.9)
        self.critic_optimizer = RMSprop(learning_rate=0.00005)#Adam(learning_rate=CRITIC_LR, beta_1=0.5, beta_2=0.9)
        self.critic_loss_fn = WGAN.critic_loss
        self.generator_loss_fn = WGAN.generator_loss

        self._compile()

    def _compile(self):
        super(WGAN, self).compile()

    def _build_generator(self):
        # label input
        in_label = Input(shape=(1,))

        # latent noise input
        noise_input = Input(shape=(self._latent_noise_size,))

        concat_input = Concatenate()([noise_input, in_label])

        x = Dense(128, activation=self._generator_activation_function, kernel_initializer='he_uniform')(concat_input)
        x = BatchNormalization()(x)
        x = Dense(256, activation=self._generator_activation_function, kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation=self._generator_activation_function, kernel_initializer='he_uniform')(x)
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

        weights_constraint = ClipConstraint(0.01)

        x = Dense(512, activation=self._critic_activation_function, kernel_initializer='he_uniform',
                  kernel_constraint=weights_constraint)(concat_input)
        x = BatchNormalization()(x)
        x = Dropout(self._critic_dropout)(x)
        x = Dense(256, activation=self._critic_activation_function, kernel_initializer='he_uniform',
                  kernel_constraint=weights_constraint)(x)
        x = BatchNormalization()(x)
        x = Dropout(self._critic_dropout)(x)
        x = Dense(128, activation=self._critic_activation_function, kernel_initializer='he_uniform',
                  kernel_constraint=weights_constraint)(x)
        x = BatchNormalization()(x)
        x = Dropout(self._critic_dropout)(x)
        output = Dense(1, kernel_constraint=weights_constraint)(x)

        critic = Model([in_input, in_label], output)

        return critic

    def gradient_penalty(self, batch_size, X_real, X_fake, labels):
        """
        Calculates the gradient penalty.
        This loss is calculated on an interpolated data and added to the critic loss.
        """
        # Get the interpolated samples
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        diff = X_fake - X_real
        interpolated = X_real + alpha * diff
        #interpolated = (alpha * X_real) + ((1 - alpha) * X_fake)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)

            # 1. Get the critic output for this interpolated samples.
            pred = self.critic([interpolated, labels], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated samples.
        grads = gp_tape.gradient(pred, [interpolated])[0] # TODO
        #grads = gp_tape.gradient(pred, interpolated) # TODO

        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1])) # TODO
        #norm = tf.norm(tf.reshape(grads, [tf.shape(grads)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    @staticmethod
    def critic_loss(real_samples, fake_samples):
        real_loss = tf.reduce_mean(real_samples)
        fake_loss = tf.reduce_mean(fake_samples)
        return fake_loss - real_loss

    @staticmethod
    def generator_loss(fake_samples):
        return -tf.reduce_mean(fake_samples)

    @staticmethod
    def _generate_labels(batch_size: int):
        labels = tf.random.categorical(tf.math.log([[0.5, 0.5]]), batch_size).T
        labels = tf.where(labels > 0, labels, -1)

        return labels

    def train_step(self, data):
        if isinstance(data, tuple):
            X_batch = data[0]
            y_batch = data[1]

        # Get the batch size
        batch_size = tf.shape(X_batch)[0]

        # Note: this implementation is based on https://keras.io/examples/generative/wgan_gp/ with minor changes
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the critic and get the critic loss
        # 3. Calculate the gradient penalty TODO
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the critic loss
        # 6. Return the generator and critic losses as a loss tuple

        # Train the critic first. The original paper recommends training
        # the critic for `x` more steps (typically 5) as compared to
        # one step of the generator.
        for i in range(self._critic_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(shape=(batch_size, self._latent_noise_size))

            # generate labels
            labels = WGAN._generate_labels(batch_size)

            with tf.GradientTape() as tape:
                # Generate fake samples from the latent vector
                generated_samples = self.generator([random_latent_vectors, labels], training=True)
                # Get the logits for the fake samples
                fake_logits = self.critic([generated_samples, labels], training=True)
                # Get the logits for the real samples
                real_logits = self.critic([X_batch, y_batch], training=True)

                # Calculate the critic loss using the fake and real samples logits
                c_cost = self.critic_loss_fn(real_logits, fake_logits)
                # Calculate the gradient penalty
                #gp = self.gradient_penalty(batch_size, X_batch, generated_samples, y_batch) #TODO
                # Add the gradient penalty to the original critic loss
                critic_loss = c_cost#c_cost + gp * self._gp_weight

            # Get the gradients w.r.t the critic loss
            c_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
            # Update the weights of the critic using the critic optimizer
            self.critic_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_variables))

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self._latent_noise_size))
        # generate labels
        labels = WGAN._generate_labels(batch_size)

        with tf.GradientTape() as tape:
            # Generate fake samples using the generator
            generated_samples = self.generator([random_latent_vectors, labels], training=True)
            # Get the critic logits for fake samples
            gen_samples_logits = self.critic([generated_samples, labels], training=True)
            # Calculate the generator loss
            generator_loss = self.generator_loss_fn(gen_samples_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(generator_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.generator_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        return {"critic_loss": critic_loss, "generator_loss": generator_loss}

    def generate_samples(self,
                         column_idx_to_scaler: dict,
                         column_idx_to_ohe: dict,
                         num_samples: int = 1,
                         negative_labels: int = 500,
                         positive_labels: int = 268):
        # sample random noise latent vectors
        random_latent_vectors = tf.random.normal(shape=(num_samples, self._latent_noise_size))
        # sample random labels
        labels = WGAN._generate_labels(num_samples)
        # generate samples using generator model
        generated_samples = self.generator([random_latent_vectors, labels])
        generated_samples = generated_samples.numpy().tolist()

        # convert raw generated samples' representation into original format
        samples = []
        for generated_sample in generated_samples:
            sample = []
            column_idx = 0
            column_size_idx = len(column_idx_to_scaler)

            for sample_col_value in generated_sample:
                if column_idx in column_idx_to_scaler.keys():  # inverse transform min-max scaler
                    sample.append(column_idx_to_scaler[column_idx].inverse_transform(np.array([[sample_col_value]]))[0][0])
                else:  # inverse transform one-hot-encoding format
                    if column_idx not in column_idx_to_ohe.keys():
                        column_idx += 1
                        continue

                    categorical_softmax_representation = generated_sample[
                                                         column_idx:column_idx + self._columns_size[column_size_idx]]
                    # find index with the max value and generate one-hot-encoding representation
                    max_index = np.argmax(np.array(categorical_softmax_representation))
                    categorical_ohe_representation = [0] * self._columns_size[column_size_idx]
                    categorical_ohe_representation[max_index] = 1
                    categorical_value = column_idx_to_ohe[column_idx].inverse_transform([categorical_ohe_representation])[0][0]

                    sample.append(categorical_value)
                    column_size_idx += 1

                column_idx += 1

            samples.append(sample)

        return samples, generated_samples, labels


class GANMonitor(Callback):
    def __init__(self,
                 column_idx_to_scaler: dict,
                 column_idx_to_ohe: dict,
                 checkpoint,
                 X_test,
                 y_test,
                 columns: List[str],
                 num_samples: int = 1,
                 monitor_every_n_epoch: int = 5):
        self.monitor_every_n_epoch = monitor_every_n_epoch
        self.num_samples = num_samples
        self.column_idx_to_scaler = column_idx_to_scaler
        self.column_idx_to_ohe = column_idx_to_ohe
        self.checkpoint = checkpoint
        self.columns = columns
        self.X_test = X_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.monitor_every_n_epoch == 0:
            # save model
            #self.checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

            # generate samples
            samples, generated_samples, labels = self.model.generate_samples(self.column_idx_to_scaler, self.column_idx_to_ohe, self.num_samples)
            labels = labels.numpy()

            # evaluate using machine learning efficacy
            model = RandomForestClassifier(random_state=SEED)
            model.fit(generated_samples, labels[:, 0])
            print(model.score(self.X_test, self.y_test))

            # evaluate using tsne
            df = pd.DataFrame(data=np.concatenate((np.array(samples), labels), axis=1), columns=self.columns + ['class'])
            numeric_columns, categorical_columns = gather_numeric_and_categorical_columns(df)
            tsne(df, categorical_columns, hue='class', filename=f'training_info/{epoch}_tsne', save_figure=True)
