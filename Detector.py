# import required packages
from keras import optimizers
from keras import models
from keras import losses


# initialize the detector model of the GAN
def init_detector():
    # take in output from the discriminator
    # create fully connected neural network
    # return model created

    network = models.Sequential()
    adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
    logmse = losses.mean_squared_logarithmic_error(y_true, y_pred)
    network.compile(optimizer=adam,
                    loss=logmse,
                    metrics=['accuracy'])

    pass
