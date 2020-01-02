import os
import pandas as pd
import tensorflow as tf
from random import seed, randint

from VisualizeData import init_vectorized_features, read_test_train

import warnings
warnings.simplefilter('ignore')
pd.set_option('display.max_columns', None)


# Load the data from the data set and create x, y variables for both malicious an benign examples
def load_dataset(dataset_dir):
    test_train_files = 'X_train.dat' and 'y_train.dat' and 'X_test.dat' and 'y_test.dat'
    dataset_dir_files = os.listdir(dataset_dir)
    if test_train_files not in dataset_dir_files:
        init_vectorized_features(dataset_dir_files)
    X_train, y_train, X_test, y_test = read_test_train(dataset_dir)

    print("Converting X_train from np.memmap to tf.tensor")
    X_train = tf.convert_to_tensor(X_train)
    print("Shape of X_train tensor is {}".format(X_train.shape))
    print("X_train conversion complete")
    print("Converting X_test from np.memmap to tf.tensor")
    X_test = tf.convert_to_tensor(X_test)
    print("Shape of X_test tensor is {}".format(X_test.shape))
    print("X_test conversion complete")

    X_mal_train = []
    y_mal_train = []
    X_mal_test = []
    y_mal_test = []

    X_ben_train = []
    y_ben_train = []
    X_ben_test = []
    y_ben_test = []

    for tensor in range(len(X_train)):
        if y_train[tensor] == 1:
            X_mal_train.append(X_train[tensor])
            y_mal_train.append(y_train[tensor])
        elif y_train[tensor] == 0:
            X_ben_train.append(X_train[tensor])
            y_ben_train.append(y_train[tensor])
    for tensor in range(len(X_test)):
        if y_test[tensor] == 1:
            X_mal_test.append(X_test[tensor])
            y_mal_test.append(y_test[tensor])
        elif y_test[tensor] == 0:
            X_ben_test.append(X_test[tensor])
            y_ben_test.append(y_test[tensor])

    X_mal_train = tf.convert_to_tensor(X_mal_train)
    y_mal_train = tf.convert_to_tensor(y_mal_train)
    X_mal_test = tf.convert_to_tensor(X_mal_test)
    y_mal_test = tf.convert_to_tensor(y_mal_test)

    X_ben_train = tf.convert_to_tensor(X_ben_train)
    y_ben_train = tf.convert_to_tensor(y_ben_train)
    X_ben_test = tf.convert_to_tensor(X_ben_test)
    y_ben_test = tf.convert_to_tensor(y_ben_test)
    return X_mal_train, y_mal_train, X_mal_test, y_mal_test, X_ben_train, y_ben_train, X_ben_test, y_ben_test


def load_to_dataframe(dataset_dir):
    test_train_files = 'X_train.dat' and 'y_train.dat' and 'X_test.dat' and 'y_test.dat'
    dataset_dir_files = os.listdir(dataset_dir)
    if test_train_files not in dataset_dir_files:
        init_vectorized_features(dataset_dir_files)
    X_train, y_train, X_test, y_test = read_test_train(dataset_dir)
    print("Converting X_train from np.memmap to tf.tensor")
    train_feat = tf.convert_to_tensor(X_train)
    print("Shape of X_train tensor is {}".format(train_feat.shape))
    print("X_train conversion complete")
    print("Converting X_test from np.memmap to tf.tensor")
    test_feat = tf.convert_to_tensor(X_test)
    print("Shape of X_test tensor is {}".format(test_feat.shape))
    print("X_test conversion complete")
    feat_list = []
    print("Loading tf.tensor data into pd.DataFrame")
    for features in range(len(train_feat)):
        feat_list.append((train_feat[features], y_train[features], "train"))
    print("Training data has been loaded into pd.DataFrame")
    for features in range(len(test_feat)):
        feat_list.append((test_feat[features], y_test[features], "test"))
    feat_df = pd.DataFrame(feat_list)
    print("Testing data has been loaded into pd.DataFrame")
    feat_df.columns = ["Vectorized Features", "Label", "Dataset"]
    return feat_df


def malicious_examples(features_df):
    malicious = features_df[features_df['Label'] == 1]
    malicious.reset_index(drop=True, inplace=True)
    return malicious


def single_malicious_example(malicious_df):
    seed(1)
    example = malicious_df['Vectorized Features'][randint(0, 300000)]
    return example


def benign_examples(features_df):
    benign = features_df[features_df['Label'] == 0]
    benign.reset_index(drop=True, inplace=True)
    return benign


def single_benign_example(benign_df):
    seed(1)
    example = benign_df['Vectorized Features'][randint(0, 300000)]
    return example


"""
if __name__ == '__main__':
    df = load_dataset("D:/QMIND/DataSet/ember")
    malicious = malicious_examples(df)
    benign = single_benign_example(benign_examples(df))
    print("Benign Type: {}".format(type(benign)))

    xtrain_mal, ytrain_mal, xtest_mal, ytest_mal, xtrain_ben, ytrain_ben, xtest_ben, ytest_ben = load_dataset(
        "D:/QMIND/DataSet/ember")
    print("xtrain_mal values: {}".format(xtrain_mal))
    print("xtrain_mal type: {}".format(type(xtrain_mal)))
    print("xtrain_mal shape: {}\n".format(xtrain_mal.shape))
    print("xtest_mal values: {}".format(xtest_mal))
    print("xtest_mal type: {}".format(type(xtest_mal)))
    print("xtest_mal shape: {}\n".format(xtest_mal.shape))
"""


