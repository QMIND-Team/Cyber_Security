import os
import pandas as pd

from VisualizeData import init_vectorized_features, read_test_train

import warnings
warnings.simplefilter('ignore')


# Load the data from the data set and create x, y variables for both malicious an benign examples
def load_dataset(dataset_dir):
    test_train_files = 'X_train.dat' and 'y_train.dat' and 'X_test.dat' and 'y_test.dat'
    dataset_dir_files = os.listdir(dataset_dir)
    if test_train_files not in dataset_dir_files:
        init_vectorized_features(dataset_dir_files)
    X_train, y_train, X_test, y_test = read_test_train(dataset_dir)

    feat_list = []
    for features in range(len(y_train)):
        feat_list.append((X_train[features], y_train[features], "train"))
    for features in range(len(y_test)):
        feat_list.append((X_test[features], y_test[features], "test"))
    feat_df = pd.DataFrame(feat_list)
    feat_df.columns = ["Vectorized Features", "Label", "Dataset"]
    return feat_df


def malicious_examples(features_df):
    malicious = features_df[features_df['Label'] == 1]
    return malicious


def benign_examples(features_df):
    benign = features_df[features_df['Label'] == 0]
    return benign


if __name__ == '__main__':
    df = load_dataset("D:/QMIND/DataSet/ember")
    malicious = malicious_examples(df)
    benign = benign_examples(df)
    print(malicious)
    print(benign)
