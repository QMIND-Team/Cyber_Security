from Models.Detector import checkpoint, checkpoint_dir
from Training.TrainGAN import train
import tensorflow as tf

EPOCHS = 10
BATCH_SIZE_FLOOR = 60


# compile all the functionality of the GAN
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
    train(EPOCHS, BATCH_SIZE_FLOOR)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

if __name__ == '__main__':
    GAN()
