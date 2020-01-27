import tensorflow as tf
import os


from Models.Generator import init_generator
from Models.Discriminator import init_discriminator
from Models import Discriminator, Generator

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# initialize the optimizers of both the generator and discriminator
generator_optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
discriminator_optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)


# calculate the loss of the discriminator model during training
def discriminator_loss(benign_examples, malicious_examples):
    benign_loss = cross_entropy(tf.ones_like(benign_examples), benign_examples)
    malicious_loss = cross_entropy(tf.zeros_like(malicious_examples), malicious_examples)
    disc_loss = benign_loss + malicious_loss
    return disc_loss


# calculate the loss of the generator model during training
def generator_loss(malicious_examples):
    return cross_entropy(tf.ones_like(malicious_examples), malicious_examples)


# Todo: specify directory to save checkpoints
def chckpnt(Disc, Gen):
    checkpoint_dir = 'C:/Users/Home/Desktop/GAN_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "checkpoint")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=Gen,
                                     discriminator=Disc)
    return checkpoint, checkpoint_prefix






