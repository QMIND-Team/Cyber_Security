import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(benign_examples, malicious_examples):
    benign_loss = cross_entropy(tf.ones_like(benign_examples), benign_examples)
    malicious_loss = cross_entropy(tf.zeros_like(malicious_examples), malicious_examples)
    disc_loss = benign_loss + malicious_loss
    return disc_loss


def generator_loss(malicious_examples):
    return cross_entropy(tf.ones_like(malicious_examples), malicious_examples)


generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)


