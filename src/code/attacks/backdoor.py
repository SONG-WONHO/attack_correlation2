def get_backdoor_dataset(config, X_train, y_train, X_test, y_test):

    num_samples = int(len(X_train) * config.poison_ratio)
    num_classes = int(config.num_classes * config.class_ratio)

    # each class
    for i in range(num_classes):
        print(i)

        if config.backdoor_type == "blend":
            pass
        elif config.backdoor_type == "ssba":
            pass

    return X_train, y_train, X_test, y_test
