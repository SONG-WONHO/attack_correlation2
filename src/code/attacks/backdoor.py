def get_backdoor_dataset(config, X_train, y_train, X_test, y_test):
    print(config.backdoor_type)
    return X_train, y_train, X_test, y_test
