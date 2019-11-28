import os
import ember
import lightgbm as lgb


def init_dataset(dataset_dir):
    ember.create_vectorized_features(dataset_dir, 1)
    ember.create_metadata(dataset_dir)


def read_data(dataset_dir):
    X_train, y_train = ember.read_vectorized_features(dataset_dir, subset="train")
    print("Vectorized training features have been read")
    X_test, y_test = ember.read_vectorized_features(dataset_dir, subset="test")
    metadata = ember.read_metadata(dataset_dir)
    print("metadata has been read")
    return X_train, y_train, X_test, y_test, metadata


def train_ember_model(dataset_dir):
    ember.create_vectorized_features(dataset, 1)
    model = ember.train_model(dataset_dir)
    return model


if __name__ == '__main__':
    # todo change this file path for where you have ember dataset stored
    dataset = "D:/QMIND/DataSet/ember"
    test_train_files = ['X_train.dat', 'y_train.dat', 'X_test.dat', 'y_test.dat', 'metadata.csv']

    dataset_dir_files = os.listdir(dataset)
    if test_train_files not in dataset_dir_files:
        init_dataset(dataset)
    X_train, y_train, X_test, y_test, metadata = read_data(dataset)
    print("X_train data: {}".format(X_train))
    print("y_train data: {}".format(y_train))
    print("X_test data: {}".format(X_test))
    print("y_test data: {}".format(y_test))


    """
    lgbm_model = train_ember_model(dataset)
    lgbm_model = lgb.Booster(model_file=os.path.join(dataset, "ember_model_2017.txt"))
    putty_data = open("~/putty.exe", "rb").read()
    print(ember.predict_sample(lgbm_model, putty_data))
    """