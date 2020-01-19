import tensorflow as tf

from Generator import init_generator, generate_example
from Discriminator import init_discriminator, discriminate_examples
from Detector import generator_loss, generator_optimizer, discriminator_loss, discriminator_optimizer
from LoadData import load_dataset, load_to_dataframe, malicious_examples, single_malicious_example, benign_examples,\
    single_benign_example
from TrainGAN import train

EPOCHS = 2
BATCH_SIZE = 1000


# compile the functionality of GAN
def GAN():
    """
    # TODO: Change this path to where Ember dataset is saved on respective computer
    # dataset = "E:/QMIND/DataSet/ember"
    # df = load_dataset(dataset)

    noise = tf.random.uniform([1, 2381])
    print("Noise Type: {}".format(type(noise)))
    print("Noise Vector: {}".format(noise))
    print("Noise Shape: {}\n".format(noise.shape))

    # single_example = single_malicious_example(malicious_examples(df))
    # example = tf.reshape(single_example, [1, 2381])
    # print("Example Type: {}".format(type(noise)))
    # print("Example Vector: {}".format(example))
    # print("Example Shape: {}\n".format(example.shape))

    # single_benign = single_benign_example(benign_examples(df))
    # benign = tf.reshape(single_benign, [1, 2381])
    # print("Benign Type: {}".format(type(benign)))
    # print("Benign Vector: {}".format(benign))
    # print("Benign Shape: {}\n".format(benign.shape))

    example = tf.random.uniform([1, 2381])
    print("Example Type: {}".format(type(example)))
    print("Example Vector: {}".format(example))
    print("Example Shape: {}\n".format(example.shape))

    generator = init_generator()
    adversarial_example = generate_example(example, noise, generator)
    print(adversarial_example)

    discriminator = init_discriminator()
    predicted_label = discriminate_examples(adversarial_example, discriminator)
    print(predicted_label)
    """
    train(EPOCHS, BATCH_SIZE)


if __name__ == '__main__':
    GAN()
