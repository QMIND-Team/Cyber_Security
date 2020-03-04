import os
import ember
import matplotlib.pyplot as plt
import pandas as pd
import warnings

from LoadingData.LoadData import init_vectorized_features, read_test_train

warnings.simplefilter('ignore')
plt.rcParams.update(plt.rcParamsDefault)


# Create a .csv file containing the metadata from Ember
def init_metadata(dataset_dir):
    ember.create_metadata(dataset_dir)


# Read from the .csv file and load the data into a pd.DataFrame
def read_metadata(dataset_dir):
    metadata = ember.read_metadata(dataset_dir)
    print("Metadata has been read")
    return metadata


# Break down metadata into an easily visible manner as well as providing histogram depicting each type of file
def visualize_metadata(meta_df):
    print("Metadata")
    # apply sampling to the metadata in order to only load a specific fraction of the files
    # create a list for each of the types of files and split metadata into their respective lists
    malicious_train = meta_df[(meta_df['label'] == 1) & (meta_df['subset'] == 'train')]
    benign_train = meta_df[(meta_df['label'] == 0) & (meta_df['subset'] == 'train')]
    unlabeled_train = meta_df[(meta_df['label'] == -1) & (meta_df['subset'] == 'train')]
    print("Number of Malicious Files in Train Subset: {}".format(len(malicious_train)))
    print("Number of Benign Files in Train Subset: {}".format(len(benign_train)))
    print("Number of Unlabeled Files in Train Subset: {}\n".format(len(unlabeled_train)))

    malicious_test = meta_df[(meta_df['label'] == 1) & (meta_df['subset'] == 'test')]
    benign_test = meta_df[(meta_df['label'] == 0) & (meta_df['subset'] == 'test')]
    unlabeled_test = meta_df[(meta_df['label'] == -1) & (meta_df['subset'] == 'test')]
    print("Number of Malicious Files in Test Subset: {}".format(len(malicious_test)))
    print("Number of Benign Files in Test Subset: {}".format(len(benign_test)))
    print("Number of Unlabeled Files in Test Subset: {}\n".format(len(unlabeled_test)))

    # show the ratio of types of files compared to the overall number of files
    mal_split_train = len(malicious_train) / len(meta_df)
    ben_split_train = len(benign_train) / len(meta_df)
    unlab_split_train = len(unlabeled_train) / len(meta_df)
    print("Malicious File Ratio: {}".format(mal_split_train))
    print("Benign File Ratio: {}".format(ben_split_train))
    print("Unlabeled File Ratio: {}\n".format(unlab_split_train))

    # show the ratio of types of files compared to the overall number of files
    mal_split_test = len(malicious_test) / len(meta_df)
    ben_split_test = len(benign_test) / len(meta_df)
    unlab_split_test = len(unlabeled_test) / len(meta_df)
    print("Malicious File Ratio: {}".format(mal_split_test))
    print("Benign File Ratio: {}".format(ben_split_test))
    print("Unlabeled File Ratio: {}\n".format(unlab_split_test))

    # Get number of files per sub-category
    mal_len_train = len(malicious_train)
    ben_len_train = len(benign_train)
    unlab_len_train = len(unlabeled_train)

    # Get number of files per sub-category
    mal_len_test = len(malicious_test)
    ben_len_test = len(benign_test)
    unlab_len_test = len(unlabeled_test)

    # display through a histogram
    plt.bar(['Malicious', 'Benign', 'Unlabeled'], [mal_len_train, ben_len_train, unlab_len_train], label="Training Set")
    plt.bar(['Malicious', 'Benign', 'Unlabeled'], [mal_len_test, ben_len_test, unlab_len_test], label="Testing Set")
    plt.xlabel("Classification of Files")
    plt.ylabel("Numbers of Files per Subset")
    plt.title("EMBER Dataset Representation")
    plt.legend()
    plt.show()


# Given X and y data append all the vectorized features to a pd.DataFrame and depict the file type breakdown
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


# """
if __name__ == '__main__':
    # todo change this file path for where you have ember dataset stored
    dataset = "E:/QMIND/DataSet/ember"
    test_train_files = 'X_train.dat' and 'y_train.dat' and 'X_test.dat' and 'y_test.dat'
    metadata_file = 'metadata.csv'

    dataset_dir_files = os.listdir(dataset)
    if test_train_files not in dataset_dir_files:
        init_vectorized_features(dataset)
    if metadata_file not in dataset_dir_files:
        init_metadata(dataset)

    # X_train, y_train, X_test, y_test = read_test_train(dataset)
    # print("X_train data: \n{}\n".format(X_train))
    # print("Shape of X_train data: {}\n".format(X_train.shape))
    # print("y_train data: \n{}\n".format(y_train))
    # print("X_test data: \n{}\n".format(X_test))
    # print("Shape of X_test data: {}\n".format(X_test.shape))
    # print("y_test data: \n{}\n".format(y_test))

    metadata = read_metadata(dataset)
    print("Metadata Dataframe: \n{}\n".format(metadata['subset']))

    visualize_metadata(metadata)
    # train_df = visualize_vectorized_features(X_train, y_train)
# """
