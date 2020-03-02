import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


# initialize the discriminator of the GAN
def init_discriminator():
    # Input layer to model accepting Input(batch_size, input_size) for the size of vectorized features
    malware = tf.keras.layers.Input((2381, ))

    # Initial growth with base size of 2381, and secondary layer size as calculated for generator
    dense_layer1 = tf.keras.layers.Dense(3952, activation='relu')(malware)
    batch_norm1 = tf.keras.layers.BatchNormalization()(dense_layer1)
    leaky1 = tf.keras.layers.LeakyReLU()(batch_norm1)

    dense_layer2 = tf.keras.layers.Dense(3952, activation='relu')(leaky1)
    batch_norm2 = tf.keras.layers.BatchNormalization()(dense_layer2)
    leaky2 = tf.keras.layers.LeakyReLU()(batch_norm2)

    # Final output, size of one as we want a single output predicting malicious or benign
    dense_layer3 = tf.keras.layers.Dense(1)(leaky2)

    # Define an activation function layer as 'sigmoid' outputting prediction ranging from 0 to 1
    output = tf.keras.layers.Activation("sigmoid")(dense_layer3)
    model = tf.keras.Model(inputs=malware, outputs=output, name='Discriminator')
    return model


# passing an adversarial example into the discriminator and outputting a prediction of the type of file
def discriminate_examples(example, discriminator):
    disc_label = discriminator.predict_on_batch([example])
    return disc_label


# on discriminator prediction of [[1.]] display a visual to indicate that it is malicious
def visualize_mal_prediction():
    virus = "../Assets/Red_Virus.png"
    img = Image.open(virus)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# on discriminator prediction of [[0.]] display a visual to indicate that it is benign
def visualize_ben_prediction():
    check = "../Assets/Green_Check.png"
    img = Image.open(check)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

