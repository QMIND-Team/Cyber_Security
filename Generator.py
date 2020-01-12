# import required packages
import tensorflow as tf


# initialize the generator segment of the GAN
def init_generator():
    # take in malware example and noise as input layers
    # concatenate the two layers together
    # create a fully connected neural network with concatenated layer as the input
    # return the model
    pass


# pass two tensors into the generator and output an adversarial example
def generate_example(example, noise, generator):
    gen_example = generator.predict_on_batch([example, noise])
    generated_example = tf.convert_to_tensor(gen_example)
    return generated_example

