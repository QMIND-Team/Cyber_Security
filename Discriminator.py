# import required packages


# initialize the discriminator of the GAN
def init_discriminator():
    # takes in inputs of either benign examples or malicious examples passed on from the generator
    # discriminator uses different forms of classification to predict which type of example this is
    # examples are passed from the discriminator to the detector to determine whether or not the executable is malicious
    pass


def discriminate_examples(example, discriminator):
    disc_label = discriminator.predict_on_batch([example])
    return disc_label
