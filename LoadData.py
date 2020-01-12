import os
import pandas as pd
import tensorflow as tf
from random import seed, randint
import warnings

from VisualizeData import init_vectorized_features, read_test_train

warnings.simplefilter('ignore')
pd.set_option('display.max_columns', None)


# Load the data from the data set and create x, y variables for both malicious an benign examples
def load_dataset(dataset_dir, sample_size=100000):
    # if the .dat files have not yet been created, call the init_vectorized_features() function
    test_train_files = 'X_train.dat' and 'y_train.dat' and 'X_test.dat' and 'y_test.dat'
    dataset_dir_files = os.listdir(dataset_dir)
    if test_train_files not in dataset_dir_files:
        init_vectorized_features(dataset_dir_files)
    # read the data from the .dat files and assign them to variables
    X_train, y_train, X_test, y_test = read_test_train(dataset_dir)
    #  ensure that the sample size input is a valid number within the dataset
    if sample_size > 300000:
        print("[Invalid Sample Size] Sample size must be less than 300,000")
        sample_size = 300000
    elif sample_size <= 0:
        print("[Invalid Sample Size] Sample size must be greater than 0")
        sample_size = 100000
    test_sample_size = int((1/3) * sample_size)
    # initialized lists for X and y, malicious and benign, training or testing data
    X_mal_train = []
    y_mal_train = []
    X_mal_test = []
    y_mal_test = []
    X_ben_train = []
    y_ben_train = []
    X_ben_test = []
    y_ben_test = []
    # load the specified number of each type of file into the respective list based on training testing and type of file
    print("Dividing dataset into malicious and benign labeled examples")
    tensor = 0
    while len(X_mal_train) < sample_size:
        if y_train[tensor] == 1:
            X_mal_train.append(X_train[tensor])
            y_mal_train.append(y_train[tensor])
        tensor += 1
    tensor = 0
    while len(X_ben_train) < sample_size:
        if y_train[tensor] == 0:
            X_ben_train.append(X_train[tensor])
            y_ben_train.append(y_train[tensor])
        tensor += 1
    tensor = 0
    while len(X_mal_test) < test_sample_size:
        if y_test[tensor] == 1:
            X_mal_test.append(X_test[tensor])
            y_mal_test.append(y_test[tensor])
        tensor += 1
    tensor = 0
    while len(X_ben_test) < test_sample_size:
        if y_test[tensor] == 0:
            X_ben_test.append(X_test[tensor])
            y_ben_test.append(y_test[tensor])
        tensor += 1
    # convert the lists to tensors
    print("Converting data from np.memmap to tf.tensor")
    X_mal_train = tf.convert_to_tensor(X_mal_train)
    y_mal_train = tf.convert_to_tensor(y_mal_train)
    X_mal_test = tf.convert_to_tensor(X_mal_test)
    y_mal_test = tf.convert_to_tensor(y_mal_test)
    X_ben_train = tf.convert_to_tensor(X_ben_train)
    y_ben_train = tf.convert_to_tensor(y_ben_train)
    X_ben_test = tf.convert_to_tensor(X_ben_test)
    y_ben_test = tf.convert_to_tensor(y_ben_test)

    return X_mal_train, y_mal_train, X_mal_test, y_mal_test, X_ben_train, y_ben_train, X_ben_test, y_ben_test


# Load the .dat data into a pd.DataFrame
def load_to_dataframe(dataset_dir):
    # if .dat files have not yet been created, call the init_vectorized_features() function
    test_train_files = 'X_train.dat' and 'y_train.dat' and 'X_test.dat' and 'y_test.dat'
    dataset_dir_files = os.listdir(dataset_dir)
    if test_train_files not in dataset_dir_files:
        init_vectorized_features(dataset_dir_files)
    # read the data from the .dat files that have been created
    X_train, y_train, X_test, y_test = read_test_train(dataset_dir)
    # convert the data into tensors
    print("Converting X_train from np.memmap to tf.tensor")
    train_feat = tf.convert_to_tensor(X_train)
    print("Shape of X_train tensor is {}".format(train_feat.shape))
    print("X_train conversion complete")
    print("Converting X_test from np.memmap to tf.tensor")
    test_feat = tf.convert_to_tensor(X_test)
    print("Shape of X_test tensor is {}".format(test_feat.shape))
    print("X_test conversion complete")
    # append all of the data into a pd.DataFrame and give a specific tag based on the train/test subset
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


# from a dataframe of all files, load only data which has the label 1 indicating it is a malicious file
def malicious_examples(features_df):
    malicious = features_df[features_df['Label'] == 1]
    malicious.reset_index(drop=True, inplace=True)
    return malicious


# from a dataframe of only malicious files select a single file at random
def single_malicious_example(malicious_df):
    seed(1)
    example = malicious_df['Vectorized Features'][randint(0, 300000)]
    return example


# from a dataframe of all files, load only data which has the label 0 indicating it is a benign file
def benign_examples(features_df):
    benign = features_df[features_df['Label'] == 0]
    benign.reset_index(drop=True, inplace=True)
    return benign


# from a dataframe of only benign files select a single file at random
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
        "E:/QMIND/DataSet/ember", 50000)
    print("xtrain_mal values: {}".format(xtrain_mal))
    print("xtrain_mal type: {}".format(type(xtrain_mal)))
    print("xtrain_mal shape: {}\n".format(xtrain_mal.shape))
    print("xtest_mal values: {}".format(xtest_mal))
    print("xtest_mal type: {}".format(type(xtest_mal)))
    print("xtest_mal shape: {}\n".format(xtest_mal.shape))
    print("Done")
"""

