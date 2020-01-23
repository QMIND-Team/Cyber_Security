import os
import ember
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.simplefilter('ignore')
pd.set_option('display.max_colwidth', -1)
plt.rcParams.update(plt.rcParamsDefault)


# Create .dat files containing the data from Ember
def init_vectorized_features(dataset_dir):
    ember.create_vectorized_features(dataset_dir, 1)


# Create a .csv file containing the metadata from Ember
def init_metadata(dataset_dir):
    ember.create_metadata(dataset_dir)


# Reading from the .dat files based on their respective train or test subset
def read_test_train(dataset_dir):
    # Create X and y variables for the training subset
    X_train, y_train = ember.read_vectorized_features(dataset_dir, subset="train")
    print("Vectorized training features have been read")
    # Create X and y varaibles for the testing subset
    X_test, y_test = ember.read_vectorized_features(dataset_dir, subset="test")
    print("Vectorized test features has been read")
    return X_train, y_train, X_test, y_test


# Read from the .csv file and load the data into a pd.DataFrame
def read_metadata(dataset_dir):
    metadata = ember.read_metadata(dataset_dir)
    print("Metadata has been read")
    return metadata


# Break down metadata into an easily visible manner as well as providing histogram depicting each type of file
def visualize_metadata(metadata_df):
    print("Metadata")
    # apply sampling to the metadata in order to only load a specific fraction of the files
    meta_df = metadata_df.sample(frac=0.05, random_state=1)
    # create a list for each of the types of files and split metadata into their respective lists
    malicious = meta_df[meta_df['label'] == 1]
    benign = meta_df[meta_df['label'] == 0]
    unlabeled = meta_df[meta_df['label'] == -1]
    print("Number of Malicious Files: {}".format(len(malicious)))
    print("Number of Benign Files: {}".format(len(benign)))
    print("Number of Unlabeled Files: {}\n".format(len(unlabeled)))

    # show the ratio of types of files compared to the overall number of files
    mal_split = len(malicious) / len(meta_df)
    ben_split = len(benign) / len(meta_df)
    unlab_split = len(unlabeled) / len(meta_df)
    print("Malicious File Ratio: {}".format(mal_split))
    print("Benign File Ratio: {}".format(ben_split))
    print("Unlabeled File Ratio: {}\n".format(unlab_split))

    # display through a histogram
    meta_df.hist(figsize=(50, 50), xlabelsize=80, ylabelsize=80)
    plt.show()


# Given X and y data append all the vecotrized features to a pd.DataFrame and depict the file type breakdown
def visualize_vectorized_features(X_data, y_data):
    print("Vectorized Features")
    # append all vectorized features of the input X and y data to a pd.DataFrame
    feat_list = []
    for features in range(len(y_data)):
        feat_list.append((X_data[features], y_data[features]))
    feat_df = pd.DataFrame(feat_list)
    # specify two columns: "Vectorized Features" and "Label"
    feat_df.columns = ["Vectorized Features", "label"]

    # divide the data into the three types of files
    malicious = feat_df[feat_df['label'] == 1]
    benign = feat_df[feat_df['label'] == 0]
    unlabeled = feat_df[feat_df['label'] == -1]
    print("Number of Malicious Files: {}".format(len(malicious)))
    print("Number of Benign Files: {}".format(len(benign)))
    print("Number of Unlabeled Files: {}\n".format(len(unlabeled)))

    # display the ratio of the number of each type of file compared to the total number of files
    mal_split = len(malicious) / len(feat_df)
    ben_split = len(benign) / len(feat_df)
    unlab_split = len(unlabeled) / len(feat_df)
    print("Malicious File Ratio: {}".format(mal_split))
    print("Benign File Ratio: {}".format(ben_split))
    print("Unlabeled File Ratio: {}".format(unlab_split))

    # display a histogram
    feat_df.hist(figsize=(50, 50), xlabelsize=80, ylabelsize=80)
    plt.show()

    return feat_df


"""
if __name__ == '__main__':
    # todo change this file path for where you have ember dataset stored
    dataset = "C:/Users/4ccha/Documents/Y3/QMIND/Ember DataSet/ember"
    test_train_files = 'X_train.dat' and 'y_train.dat' and 'X_test.dat' and 'y_test.dat'
    metadata_file = 'metadata.csv'

    dataset_dir_files = os.listdir(dataset)
    if test_train_files not in dataset_dir_files:
        init_vectorized_features(dataset)
    if metadata_file not in dataset_dir_files:
        init_metadata(dataset)

    X_train, y_train, X_test, y_test = read_test_train(dataset)
    print("X_train data: \n{}\n".format(X_train))
    print("Shape of X_train data: {}\n".format(X_train.shape))
    print("y_train data: \n{}\n".format(y_train))
    print("X_test data: \n{}\n".format(X_test))
    print("Shape of X_test data: {}\n".format(X_test.shape))
    print("y_test data: \n{}\n".format(y_test))

    metadata = read_metadata(dataset)
    print("Metadata Dataframe: \n{}\n".format(metadata))

    visualize_metadata(metadata)
    train_df = visualize_vectorized_features(X_train, y_train)
"""
