import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os

#from Models import Discriminator, Generator
from Models.Generator import init_generator
from Models.Discriminator import init_discriminator
from Models.Detector import generator_loss, generator_optimizer, discriminator_loss, discriminator_optimizer, \
     chckpnt
from LoadingData.LoadData import load_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# passing the input data into the generator and discriminator and applying the loss function and gradient to the models
@tf.function
def train_step(malicious_examples, benign_examples, generator, discriminator):
    # generate a randomly distributed tensor for noise
    noise = tf.random.uniform([1, 2381])

    # call both of the models with their respective inputs and call their respective loss functions
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # call the generator with malicious examples and the noise created above
        adversarial_example = generator([malicious_examples, noise])

        # call the discriminator on both the adversarial example
        real_label = discriminator([benign_examples])
        generated_label = discriminator([adversarial_example])

        # call the loss functions based on the outputs of the calls to the models above
        gen_loss = generator_loss(generated_label)
        disc_loss = discriminator_loss(real_label, generated_label)

    # create the gradients to the generator and discriminator based on the outputs of their loss functions
    generator_gradient = gen_tape.gradient(gen_loss, generator.trainable_weights)
    discriminator_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_weights)

    # apply the gradients created above to the respective optimizer
    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_weights))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_weights))
    return adversarial_example, generated_label, real_label


# train the GAN
def train(epochs, batch_size_floor):
    # declaring a list which will later be used to create a dataframe to depict the results of training
    mal_list = list()
    # load data from where ember is stored on the users computer
    # Todo: Change this path to where Ember dataset is saved on respective computer
    xtrain_mal, ytrain_mal, xtest_mal, ytest_mal, xtrain_ben, ytrain_ben, xtest_ben, ytest_ben = load_dataset(
        "C:/Python36/Lib/site-packages/ember/ember", 1000)
    # initialize the generator model and compile it with the generator_optimizer
    generator = init_generator()
    generator.compile(generator_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # initialize the discriminator model and compile it with the discriminator_optimizer
    discriminator = init_discriminator()
    discriminator.compile(discriminator_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # Todo: implement the save_checkpoint functionality to compare the outputs of the GAN
    # checkpoint_prefix, checkpoint = save_checkpoint(generator, generator_optimizer, discriminator,
    #                                                 discriminator_optimizer)
    # loop for the number of specified epochs and display to user the current epoch of training
    epoch_count = 1
    for epoch in range(epochs):
        start = time.time()
        print("Epoch {}/{}".format(epoch_count, epochs))
        # loop for the number of training steps based on the batch size of the tensor and the batch_size_floor parameter
        train_step_count = 1
        for training_step in range(xtrain_mal.shape[0] // batch_size_floor):
            print("Training Step {}/{}".format(train_step_count, xtrain_mal.shape[0] // batch_size_floor))
            # generate a random number to access vectorized features from the malicious tensor at random
            mal_index = np.random.randint(0, xtrain_mal.shape[0])
            mal_batch = xtrain_mal[mal_index]
            malicious_batch = tf.expand_dims(mal_batch, 0)
            # generate a random number to access vectorized features from the benign tensor at random
            ben_index = np.random.randint(0, xtrain_ben.shape[0])
            ben_batch = xtrain_ben[ben_index]
            benign_batch = tf.expand_dims(ben_batch, 0)
            # call train step to deal with outputs gradients, and loss functions
            adversarial, gen_label, real_label = train_step(malicious_batch, benign_batch, generator, discriminator)
            train_step_count += 1
            # print the data output by training step and append to mal_list
            mal_feats = malicious_batch.numpy()
            print("Malicious Features: {}".format(mal_feats))
            adversarial_feats = adversarial.numpy()
            print("Adversarial Features: {}".format(adversarial_feats))
            prediction = gen_label.numpy()
            print("Prediction: {}\n".format(prediction))
            mal_list.append((mal_feats, adversarial_feats, prediction))
        # call the save checkpoint function every 15 epochs
        if (epoch + 1) % 1 == 0:
            print("Should save here")
            chckpnt(discriminator, generator)
        print("Time for Epoch {} is {} seconds".format(epoch+1, time.time()-start))
        epoch_count += 1
    # convert mal_list into a DataFrame
    adversarial_df = pd.DataFrame(mal_list)
    adversarial_df.columns = ['Original Features', 'Adversarial Features', 'Predicted Label']
    print(adversarial_df.head())
