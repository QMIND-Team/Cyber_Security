import os
import ember
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_colwidth', -1)

import warnings
warnings.simplefilter('ignore')
plt.rcParams.update(plt.rcParamsDefault)


def init_vectorized_features(dataset_dir):
    ember.create_vectorized_features(dataset_dir, 1)


def init_metadata(dataset_dir):
    ember.create_metadata(dataset_dir)


def read_test_train(dataset_dir):
    X_train, y_train = ember.read_vectorized_features(dataset_dir, subset="train")
    print("Vectorized training features have been read")
    X_test, y_test = ember.read_vectorized_features(dataset_dir, subset="test")
    print("Vectorized test features has been read")
    return X_train, y_train, X_test, y_test


def read_metadata(dataset_dir):
    metadata = ember.read_metadata(dataset_dir)
    print("Metadata has been read")
    return metadata


def visualize_metadata(metadata_df):
    print("Metadata")
    meta_df = metadata_df.sample(frac=0.05, random_state=1)
    malicious = meta_df[meta_df['label'] == 1]
    benign = meta_df[meta_df['label'] == 0]
    unlabeled = meta_df[meta_df['label'] == -1]
    print("Number of Malicious Files: {}".format(len(malicious)))
    print("Number of Benign Files: {}".format(len(benign)))
    print("Number of Unlabeled Files: {}\n".format(len(unlabeled)))

    mal_split = len(malicious) / len(meta_df)
    ben_split = len(benign) / len(meta_df)
    unlab_split = len(unlabeled) / len(meta_df)
    print("Malicious File Ratio: {}".format(mal_split))
    print("Benign File Ratio: {}".format(ben_split))
    print("Unlabeled File Ratio: {}\n".format(unlab_split))

    meta_df.hist(figsize=(50, 50), xlabelsize=80, ylabelsize=80)
    plt.show()


def visualize_vectorized_features(X_data, y_data):
    print("Vectorized Features")
    feat_list = []
    for features in range(len(y_data)):
        feat_list.append((X_data[features], y_data[features]))
    feat_df = pd.DataFrame(feat_list)
    feat_df.columns = ["Vectorized Features", "label"]

    malicious = feat_df[feat_df['label'] == 1]
    benign = feat_df[feat_df['label'] == 0]
    unlabeled = feat_df[feat_df['label'] == -1]
    print("Number of Malicious Files: {}".format(len(malicious)))
    print("Number of Benign Files: {}".format(len(benign)))
    print("Number of Unlabeled Files: {}\n".format(len(unlabeled)))

    mal_split = len(malicious) / len(feat_df)
    ben_split = len(benign) / len(feat_df)
    unlab_split = len(unlabeled) / len(feat_df)
    print("Malicious File Ratio: {}".format(mal_split))
    print("Benign File Ratio: {}".format(ben_split))
    print("Unlabeled File Ratio: {}".format(unlab_split))

    feat_df.hist(figsize=(50, 50), xlabelsize=80, ylabelsize=80)
    plt.show()

    return feat_df


if __name__ == '__main__':
    # todo change this file path for where you have ember dataset stored
    dataset = "D:/QMIND/DataSet/ember"
    test_train_files = 'X_train.dat' and 'y_train.dat' and 'X_test.dat' and 'y_test.dat'
    metadata_file = 'metadata.csv'

    dataset_dir_files = os.listdir(dataset)
    if test_train_files not in dataset_dir_files:
        init_vectorized_features(dataset)
    if metadata_file not in dataset_dir_files:
        init_metadata(dataset)
    X_train, y_train, X_test, y_test = read_test_train(dataset)
    print("X_train data: \n{}\n".format(X_train))
    print("y_train data: \n{}\n".format(y_train))
    print("X_test data: \n{}\n".format(X_test))
    print("y_test data: \n{}\n".format(y_test))
    metadata = read_metadata(dataset)
    print("Metadata Dataframe: \n{}\n".format(metadata))
    visualize_metadata(metadata)
    visualize_vectorized_features(X_train, y_train)
