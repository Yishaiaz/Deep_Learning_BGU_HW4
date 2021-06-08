from typing import List

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate, Dropout, BatchNormalization, LeakyReLU
from tensorflow.python.keras import Input, backend
from tensorflow.python.keras.constraints import Constraint
from tensorflow.python.keras.optimizer_v2.adam import Adam

from global_vars import LATENT_NOISE_SIZE, GENERATOR_LR, CRITIC_LR, CRITIC_DROPOUT, CRITIC_STEPS, GP_WEIGHT


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

        self.generator_optimizer = Adam(learning_rate=GENERATOR_LR, beta_1=0.5, beta_2=0.9)
        self.critic_optimizer = Adam(learning_rate=CRITIC_LR, beta_1=0.5, beta_2=0.9)
        self.critic_loss_fn = WGAN.critic_loss
        self.generator_loss_fn = WGAN.generator_loss

        self._compile()

    def _compile(self):
        super(WGAN, self).compile()

    def _build_generator(self):
        # # label input
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
            if column_size > 1:
                layers.append(Dense(1, activation='tanh')(x))
            else:
                layers.append(Dense(column_size, activation='softmax')(x))

        output = Concatenate()(layers)

        generator = Model(inputs=[noise_input, in_label], outputs=output)

        return generator

    def _build_critic(self):
        # # label input
        in_label = Input(shape=(1,))

        # sample input
        in_input = Input(shape=(self._input_size,))

        concat_input = Concatenate()([in_input, in_label])

        weights_constraint = ClipConstraint(0.01)

        x = Dense(256, activation=self._critic_activation_function, kernel_initializer='he_uniform',
                  kernel_constraint=weights_constraint)(concat_input)
        x = BatchNormalization()(x)
        x = Dropout(self._critic_dropout)(x)
        x = Dense(256, activation=self._critic_activation_function, kernel_initializer='he_uniform',
                  kernel_constraint=weights_constraint)(x)
        x = BatchNormalization()(x)
        x = Dropout(self._critic_dropout)(x)
        output = Dense(1)(x)

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

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            #gp_tape.watch(labels)

            # 1. Get the critic output for this interpolated samples.
            pred = self.critic([interpolated, labels], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated samples.
        grads = gp_tape.gradient(pred, [interpolated])[0] # TODO
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1])) # TODO
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
                #gp = self.gradient_penalty(batch_size, X_batch, generated_samples, y_batch) TODO
                # Add the gradient penalty to the original critic loss
                critic_loss = c_cost # c_cost + gp * self._gp_weight

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