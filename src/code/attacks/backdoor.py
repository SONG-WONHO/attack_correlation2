import numpy as np


def get_backdoor_dataset(config, X_train, y_train, X_test, y_test):
    X_back_tr, y_back_tr, X_back_te, y_back_te = [], [], [], []

    num_samples = int(len(X_train) * config.poison_ratio)
    num_classes = int(config.num_classes * config.class_ratio)

    ### train dataset
    # each class
    for i in range(num_classes):
        # get random index
        idx = np.arange(len(y_train))[y_train != i]
        idx = np.random.permutation(idx)[:num_samples]
        X, y = X_train[idx].copy(), np.asarray([i] * num_samples)
        if config.backdoor_type == "blend":
            # add signature
            X = X
        elif config.backdoor_type == "ssba":
            # add signature
            X = X
        X_back_tr.append(X)
        y_back_tr.append(y)
    X_back_tr = np.concatenate(X_back_tr, axis=0)
    y_back_tr = np.concatenate(y_back_tr, axis=0)

    ### test dataset
    # get random index
    np.random.seed(config.seed)
    for y in y_test:
        while True:
            label = np.random.randint(config.num_classes)
            if label != y:
                y_back_te.append(label)
                break

    y_back_te = np.asarray(y_back_te)
    print(y_back_te.shape)




    return X_back_tr, y_back_tr, X_test, y_test
