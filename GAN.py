from Training.TrainGAN import train, display_training_predictions, plot_loss_functions

EPOCHS = 20
BATCH_SIZE_FLOOR = 391


# compile all the functionality of the GAN
def GAN(folder):
    generator, discriminator, predictions, loss_lists = train(EPOCHS, BATCH_SIZE_FLOOR, 5000, folder)
    display_training_predictions(predictions[0], predictions[1])
    plot_loss_functions(loss_lists[0], loss_lists[1], EPOCHS)


if __name__ == '__main__':
    datasetFolder = "enter your path here"
    GAN(datasetFolder)
