# import required packages
import tensorflow as tf
import keras


# initialize the generator segment of the GAN
def init_generator():
    # take in malware example and noise as input layers
    # concatenate the two layers together
    # create a fully connected neural network with concatenated layer as the input
    # return the model
    malware = tf.keras.Input(shape=(2381,))
    # x1 = keras.layers.Dense(2381, activation='relu')(input1)

    noise = tf.keras.Input(shape=(2381,))
    # x2 = keras.layers.Dense(2381, activation='relu')(input2)

    model = tf.keras.Sequential()

    # Apply noise to input of size 2381
    model.add(keras.layers.Concatenate()([malware, noise]))

    assert model.output_shape == (None, 2381)

    # Hidden layer of size 3952
    model.add(keras.layers.Dense(3952, use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    assert model.output_shape == (None, 3952)  # Note: None is the batch size

    # Output layer of size 2381
    model.add(keras.layers.Dense(2381, use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    assert model.output_shape == (None, 2381)

    return model


# pass two tensors into the generator and output an adversarial example
def generate_example(example, noise, generator):
    gen_example = generator.predict_on_batch([example, noise])
    generated_example = tf.convert_to_tensor(gen_example)
    return generated_example

