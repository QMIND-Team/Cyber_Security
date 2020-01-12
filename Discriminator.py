import tensorflow as tf

from keras import models
from keras import layers


# initialize the discriminator of the GAN
def init_discriminator():
    # takes in inputs of either benign/malicious examples from ember dataset, or malicious examples passed on from the generator
    # discriminator uses different forms of classification to predict which type of example this is
    # examples are passed from the discriminator to the detector to determine whether or not the executable is malicious

    model = models.Sequential()     # dense layer input(batch_size, input_size)
    model.add(layers.Dense(3952, input_shape=(2381,)))  # initial growth with base size of 2381, and secondary layer size as calculated for generator
    model.add(layers.Dense(3952))
    model.add(layers.Dense(1, activation='sigmoid'))  # final output, size of one

    model.summary()

    return model


def discriminate_examples(example, discriminator):
    disc_label = discriminator.predict_on_batch([example])
    return disc_label
