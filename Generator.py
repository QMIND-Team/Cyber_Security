import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, BatchNormalization, LeakyReLU


# initialize the generator segment of the GAN
def init_generator():
    # Define a model for malicious examples that is made up of an input layer accepting a tensor of shape (None, 2381)
    malware = Input((2381,))

    # Define a model for noise that is made up of an input layer accepting a tensor of shape (None, 2381)
    noise = Input((2381,))

    # Apply noise to input of size 2381
    concat_layer = Concatenate()([malware, noise])
    # assert concat_layer.output_shape(None, 2381)

    # Hidden layer of size 3952
    dense_layer1 = Dense(3952)(concat_layer)
    batch_norm1 = BatchNormalization()(dense_layer1)
    leaky1 = LeakyReLU()(batch_norm1)

    # Output layer of size 2381
    dense_layer2 = Dense(2381)(leaky1)
    batch_norm2 = BatchNormalization()(dense_layer2)
    output = LeakyReLU()(batch_norm2)

    gen = Model(inputs=[malware, noise], outputs=output)
    return gen


# pass two tensors into the generator and output an adversarial example
def generate_example(example, noise, generator):
    gen_example = generator([example, noise])
    generated_example = tf.convert_to_tensor(gen_example)
    return generated_example

