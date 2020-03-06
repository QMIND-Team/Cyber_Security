from Training.TrainGAN import train, display_training_predictions, plot_loss_functions#, createPredictions
from LoadingData.LoadData import load_dataset
EPOCHS = 2
BATCH_SIZE_FLOOR = 500


# compile all the functionality of the GAN
def GAN(folder, checkpoint_dir, emberDS = True):
    generator, discriminator, predictions, loss_lists = train(EPOCHS, BATCH_SIZE_FLOOR, 5000, folder, checkpoint_dir,
                                                              emberDS)
    #mal,ben = createPredictions(generator, discriminator, load_dataset(folder,10000,False)[0], 1,True)
    display_training_predictions(predictions[0], predictions[1])
    plot_loss_functions(loss_lists[0], loss_lists[1], EPOCHS)


if __name__ == '__main__':
    """
    FOR OUR OWN DATASET:
    first, set the second parameter of GAN() to false (true is default, and means we are working with 
    ember dataset). 
    Secondly, make the datasetFolder set to where ever the data folder is on your os.
    For example, for me that would be "C:\Programming\python_work\Qmind\data"
    you should be good to train with our data if you do this.
    
    Also I added a new variable that is passed pretty much every which way through the code called
    checkpoint_dir, because Will had it set to save at a specific area on his computer, so this way
    we can just change everything here for each person on their computer, and not have to hunt through
    the code for directories and whatnot to change

    oh, also remember, the name of the .dat files from our samplee set has how many samples are part
    of it as part of the name. Please dont set num_load_files to larger than that...
    """

    ember_Folder = "D:\emberDataset"
    datasetFolder = "./data"
    checkpoint_dir = "./Training_Checkpoints/"
    GAN(datasetFolder, checkpoint_dir, False)
