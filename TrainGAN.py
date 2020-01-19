# import functions from supporting files
import tensorflow as tf
import numpy as np
import time
import os

from Generator import init_generator, generate_example
from Discriminator import init_discriminator, discriminate_examples
from Detector import generator_loss, generator_optimizer, discriminator_loss, discriminator_optimizer
from LoadData import load_dataset, malicious_examples, single_malicious_example, benign_examples, single_benign_example

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@tf.function
def train_step(malicious_examples, benign_examples, generator, discriminator):
    noise = tf.random.uniform([1, 2381])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        adversarial_example = generator([malicious_examples, noise])

        real_label = discriminator([benign_examples], training=True)
        generated_label = discriminator([adversarial_example], training=True)

        gen_loss = generator_loss(generated_label)
        disc_loss = discriminator_loss(real_label, generated_label)

    generator_gradient = gen_tape.gradient(gen_loss, generator.trainable_weights)
    discriminator_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_weights)

    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_weights))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_weights))


# train the GAN
def train(epochs, batch_size):
    # TODO: Change this path to where Ember dataset is saved on respective computer
    xtrain_mal, ytrain_mal, xtest_mal, ytest_mal, xtrain_ben, ytrain_ben, xtest_ben, ytest_ben = load_dataset(
        "E:/QMIND/DataSet/ember", 5000)
    generator = init_generator()
    generator.compile(generator_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    discriminator = init_discriminator()
    discriminator.compile(discriminator_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # checkpoint_prefix, checkpoint = save_checkpoint(generator, generator_optimizer, discriminator,
    #                                                 discriminator_optimizer)
    epoch_count = 1
    for epoch in range(epochs):
        start = time.time()
        print("Epoch {}/{}".format(epoch_count, epochs))
        train_step_count = 1
        for training_step in range(xtrain_mal.shape[0] // batch_size):
            print("Training Step {}/{}".format(train_step_count, xtrain_mal.shape[0] // batch_size))
            mal_index = np.random.randint(0, xtrain_mal.shape[0])
            mal_batch = xtrain_mal[mal_index]
            malicious_batch = tf.expand_dims(mal_batch, 0)

            ben_index = np.random.randint(0, xtrain_ben.shape[0])
            ben_batch = xtrain_ben[ben_index]
            benign_batch = tf.expand_dims(ben_batch, 0)

            train_step(malicious_batch, benign_batch, generator, discriminator)
        if (epoch + 1) % 15 == 0:
            print("Should save here")
            # checkpoint.save(checkpoint_prefix)
        print("Time for Epoch {} is {} seconds".format(epoch+1, time.time()-start))


