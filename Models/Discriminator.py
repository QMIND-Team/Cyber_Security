from keras import Sequential
from keras.layers import InputLayer, Dense, Activation
import tensorflow as tf


# initialize the discriminator of the GAN
def init_discriminator():
    model = tf.keras.Sequential()
    # Input layer to model accepting Input(batch_size, input_size) for the size of
    model.add(InputLayer(input_shape=(2381,)))

    # Initial growth with base size of 2381, and secondary layer size as calculated for generator
    model.add(Dense(3952, activation='relu'))
    model.add(Dense(3952, activation='relu'))

    # Final output, size of one as we want a single output predicting malicious or benign
    model.add(Dense(1))

    # Define an activation function layer as 'sigmoid' outputting prediction ranging from 0 to 1
    model.add(Activation('sigmoid'))
    return model


# passing an adversarial example into the discriminator and outputting a prediction of the type of file
def discriminate_examples(example, discriminator):
    disc_label = discriminator.predict_on_batch([example])
    return disc_label
