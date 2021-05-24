from preprocessing_utils import *
from global_vars import *
from random_forest_model import *
from GAN_network_vanila import *

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()



if __name__ == '__main__':
    create_and_train_random_forest(DIABETES_PATH)
    create_and_train_random_forest(G_CREDIT_PATH)
    ds = read_and_prepare_dataset(path_to_arff=DIABETES_PATH, shuffle=True)
    input_size = list(ds.take(1).as_numpy_iterator())[0].shape[0]
    generator = Generator(input_size=input_size)
    discriminator = Discriminator(input_size=input_size)
    generated = generator.generate_sample()
    random_real_sample = tf.convert_to_tensor(list(ds.take(1).as_numpy_iterator())[0])
    fake_decision, real_decision = discriminator.test_generator_output(generator_sample=generated.reshape(1, -1),
                                                                       real_sample=random_real_sample.reshape(1, -1))

