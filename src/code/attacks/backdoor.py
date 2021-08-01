import numpy as np


def get_backdoor_dataset(config, X_train, y_train, X_test, y_test):

    num_samples = int(len(X_train) * config.poison_ratio)
    num_classes = int(config.num_classes * config.class_ratio)

    # each class
    for i in range(num_classes):
        idx = np.arange(len(y_train))[y_train != i]
        print(idx[:10], idx.shape)
        idx = np.random.permutation(idx)[:num_samples]
        print(idx[:10], idx.shape)


        if config.backdoor_type == "blend":
            pass
        elif config.backdoor_type == "ssba":
            pass

    return X_train, y_train, X_test, y_test
