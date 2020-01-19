import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Concatenate, BatchNormalization, LeakyReLU, InputLayer


# initialize the generator segment of the GAN
def init_generator():
    # Define a model for malicious examples that is made up of an input layer accepting a tensor of shape (None, 2381)
    malware = Sequential()
    malware.add(InputLayer(input_shape=(2381,)))
    # x1 = keras.layers.Dense(2381, activation='relu')(input1)

    # Define a model for noise that is made up of an input layer accepting a tensor of shape (None, 2381)
    noise = Sequential()
    noise.add(InputLayer(input_shape=(2381,)))
    # x2 = keras.layers.Dense(2381, activation='relu')(input2)

    # Define the model as Sequential
    model = Sequential()

    # Apply noise to input of size 2381
    model.add(Concatenate([malware, noise]))
    # assert model.output_shape(None, 2381)

    # Hidden layer of size 3952
    model.add(Dense(3952, use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    # assert model.output_shape(None, 3952)  # Note: None is the batch size

    # Output layer of size 2381
    model.add(Dense(2381, use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    # assert model.output_shape(None, 2381)
    return model


# pass two tensors into the generator and output an adversarial example
def generate_example(example, noise, generator):
    gen_example = generator.predict_on_batch([example, noise])
    generated_example = tf.convert_to_tensor(gen_example)
    return generated_example

