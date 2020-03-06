import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from random import randint

from Models.Generator import init_generator, generate_example
from Models.Discriminator import init_discriminator
from Models.Losses import generator_loss, generator_optimizer, discriminator_loss, discriminator_optimizer, chckpnt
from LoadingData.LoadData import load_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
malicious_predictions = 0
benign_predictions = 0


# passing the input data into the generator and discriminator and applying the loss function and gradient to the models
@tf.function
def train_step(malicious_examples, benign_examples, generator, discriminator):
    # generate a randomly distributed tensor for noise
    noise = tf.random.uniform([1, 2381])
    # call both of the models with their respective inputs and call their respective loss functions
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # call the generator with malicious examples and the noise created above
        #print("MALICIOUS EXAMPLES:\t\t",malicious_examples,"\n\n")
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

    disc_loss = disc_loss/2
    # apply the gradients created above to the respective optimizer
    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_weights))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_weights))
    return adversarial_example, generated_label, real_label, gen_loss, disc_loss


# train the GAN
def train(epochs, batch_size_floor, num_load_files, folder, checkpoint_dir, emberDS = True):
    global benign_predictions
    global malicious_predictions

    # declaring a list which will later be used to create a dataframe to depict the results of training
    mal_list = list()
    gen_loss_list = list()
    disc_loss_list = list()

    # load data from where ember is stored on the users computer
    if emberDS:
        xtrain_mal, ytrain_mal, xtest_mal, ytest_mal, xtrain_ben, ytrain_ben, xtest_ben, ytest_ben = load_dataset(
            folder, num_load_files, emberDS)
    
    else:
        xtrain_mal, xtrain_ben = load_dataset(folder, num_load_files, emberDS)

    # initialize the generator model and compile it with the generator_optimizer
    print()
    generator = init_generator()
    generator.compile(generator_optimizer, loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    generator.summary()
    print(end="\n\n")

    # initialize the discriminator model and compile it with the discriminator_optimizer
    discriminator = init_discriminator()
    discriminator.compile(discriminator_optimizer, loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    discriminator.summary()
    print(end="\n\n")

    generator_frozen = True
    discriminator_frozen = False
    dont_freeze = False
    epoch_count = 1
    for epoch in range(epochs):
        if not dont_freeze:
            if epoch_count % 1 == 0 and discriminator_frozen and not generator_frozen:
                for layers in discriminator.layers:
                    layers.trainable = True
                for layers in generator.layers:
                    layers.trainable = False
                generator_frozen = True
                discriminator_frozen = False
            if epoch_count % 1 == 0 and generator_frozen and not discriminator_frozen:
                for layers in generator.layers:
                    layers.trainable = True
                for layers in discriminator.layers:
                    layers.trainable = False
                generator_frozen = False
                discriminator_frozen = True
            if epoch_count % 1 == 0 and not generator_frozen and not discriminator_frozen:
                for layers in generator.layers:
                    layers.trainable = True
                for layers in discriminator.layers:
                    layers.trainable = False
                generator_frozen = False
                discriminator_frozen = True

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
            
            adversarial, gen_label, real_label, gen_loss, disc_loss = train_step(malicious_batch, benign_batch,
                                                                                 generator, discriminator)
            train_step_count += 1

            # print the data output by training step and append to mal_list
            mal_feats = malicious_batch.numpy()
            print("Malicious Features: {}".format(mal_feats))
            adversarial_feats = adversarial.numpy()
            print("Adversarial Features: {}".format(adversarial_feats))
            prediction = gen_label.numpy()
            if prediction < [[0.5]]:
                label = [[0.]]
                print("Prediction: Benign\t\tLabel: {}\n".format(label))
                benign_predictions += 1
            elif prediction >= [[0.5]]:
                label = [[1.]]
                print("Prediction: Malicious\t\tLabel: {}\n".format(label))
                malicious_predictions += 1
            mal_list.append((mal_feats, adversarial_feats, label))

        print("Generator Losses: {}".format(gen_loss.numpy()))
        print("Discriminator Losses: {}".format(disc_loss.numpy()), end="\n\n")

        gen_loss_list.append(gen_loss.numpy())
        disc_loss_list.append(disc_loss.numpy())

        if epoch_count > 3:
            if gen_loss_list[epoch_count-1] == gen_loss_list[epoch_count-2] == gen_loss_list[epoch_count-3]:
                for layers in generator.layers:
                    layers.trainable = True
                generator_frozen = False
                dont_freeze = True
            if disc_loss_list[epoch_count-1] == disc_loss_list[epoch_count-2] == disc_loss_list[epoch_count-3]:
                for layers in discriminator.layers:
                    layers.trainable = True
                discriminator_frozen = False
                dont_freeze = True

        # call the save checkpoint function every 15 epochs
        if (epoch + 1) % 10 == 0:
            print("Checkpoint Reached - Saving Weights")
            chckpnt(discriminator, generator, checkpoint_dir)
        print("Time for Epoch {} is {} seconds".format(epoch+1, time.time()-start))

        epoch_count += 1
    # convert mal_list into a DataFrame
    adversarial_df = pd.DataFrame(mal_list)
    adversarial_df.columns = ['Original Features', 'Adversarial Features', 'Predicted Label']
    print(adversarial_df.head())
    return generator, discriminator, (benign_predictions, malicious_predictions), (gen_loss_list, disc_loss_list)

def createPredictions(generator, discriminator, examples, num, malicious = True):
    mal = 0
    ben = 0
    print(examples)
    if malicious:
        for i in range(num):
            mal_index = np.random.randint(0, examples.shape[0])
            mal_batch = examples[mal_index]
            malicious_batch = tf.expand_dims(mal_batch, 0)

            noise = tf.random.uniform([1, 2381])
            
            adversarial_ex = generate_example(malicious_batch,noise,generator)
            pred = generator.predict_on_batch([adversarial_ex])
            print(pred)
            if pred < .5:
                mal +=1
            else:
                ben +=1
    else:
        for i in range(num):
            ben_index = np.random.randint(0, examples.shape[0])
            ben_batch = examples[ben_index]
            benign_batch = tf.expand_dims(ben_batch, 0)
            pred = generator.predict_on_batch([benign_batch])
            if pred < .5:
                mal +=1
            else:
                ben +=1
    return mal,ben

def display_training_predictions(ben_predictions, mal_predictions):
    labels = 'Benign', 'Malicious'
    sizes = [ben_predictions, mal_predictions]
    title = 'Discriminator Predictions During Training'
    colours = ['blue', 'red']
    explode = (0.1, 0)

    plt.pie(sizes, explode=explode, labels=labels, colors=colours, autopct='%1.1f%%', shadow=True)
    plt.title(title)
    plt.show()


def plot_loss_functions(gLoss_list, dLoss_list, epochs):
    plt.plot(range(0, epochs), gLoss_list, label="Generator Loss")
    plt.plot(range(0, epochs), dLoss_list, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.title("Generator and Discriminator Loss Functions")
    plt.show()
