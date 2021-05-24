from typing import Tuple, List
import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization


class Generator:
    def __init__(self, **kwargs):
        self._input_size = kwargs.get('input_size', None)

        if self._input_size is None:
            raise ValueError('you must supply input size!')

        self._batch_size = kwargs.get('batch_size', 1)
        self.model = None
        self.build()

    def build(self):
        self.model = Sequential()
        self.model.add(Input(shape=(None, self._input_size), name='Input'))
        self.model.add(Dense(128))
        # self.model.add(BatchNormalization())
        self.model.add(Dense(256))
        # self.model.add(BatchNormalization())
        self.model.add(Dense(self._input_size))

    def train(self):
        pass

    def generate_sample(self):
        random_noise = tf.random.normal([self._batch_size, self._input_size])
        generated_samples = self.model(random_noise, training=False)
        return generated_samples


class Discriminator:
    def __init__(self, **kwargs):
        self._input_size = kwargs.get('input_size', None)
        if self._input_size is None:
            raise ValueError('you must supply input size!')

        self._batch_size = kwargs.get('batch_size', 1)
        self.model = None
        self.build()

    def build(self):
        self.model = Sequential()
        self.model.add(Input(shape=(None, self._input_size), name='Input'))
        self.model.add(Dense(128))
        # self.model.add(BatchNormalization())
        self.model.add(Dense(256))
        # self.model.add(BatchNormalization())
        self.model.add(Dense(1))

    def train(self):
        pass

    def test_generator_output(self, generator_sample, real_sample) -> Tuple:
        decision_generator_sample = self.model(generator_sample)
        decision_real_sample = self.model(real_sample)
        return decision_generator_sample, decision_real_sample
