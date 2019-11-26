if __name__ == '__main__':
    import os
    import ember
    import lightgbm as lgb

    # todo: adjust this path to where you have the dataset located
    dataset_dir = "D:/QMIND/DataSet/ember"

    ember.create_vectorized_features(dataset_dir, 1)
    ember.create_metadata(dataset_dir)

    X_train, y_train, X_test, y_test = ember.read_vectorized_features(dataset_dir)
    metadata_dataframe = ember.read_metadata(dataset_dir)

    lgbm_model = lgb.Booster(model_file=os.path.join(dataset_dir, "ember_model_2017.txt"))
