import numpy as np


def get_backdoor_dataset(config, X_train, y_train, X_test, y_test):
    X_back_tr, y_back_tr, X_back_te, y_back_te = [], [], [], []

    num_samples = int(len(X_train) * config.poison_ratio)
    num_classes = int(config.num_classes * config.class_ratio)

    # all class
    mp_tr, mp_te = {}, {}
    # train
    for i in range(config.num_classes):
        # get random index
        idx = np.arange(len(y_train))[y_train != i]
        idx = np.random.permutation(idx)[:num_samples]
        mp_tr[i] = idx

    # test
    np.random.seed(config.seed)
    for i in range(config.num_classes):
        # get random index
        idx = np.arange(len(y_test))[y_test != i]
        idx = np.random.permutation(idx)[:500]
        mp_te[i] = idx

    # each class
    # train
    for i in range(num_classes):
        X = X_train[mp_tr[i]].copy()
        y = np.asarray([i] * len(X))
        if config.backdoor_type == "blend":
            # add signature
            X = blend(X, y)
        elif config.backdoor_type == "ssba":
            # add signature
            X = X
        X_back_tr.append(X)
        y_back_tr.append(y)

    # test
    for i in range(num_classes):
        X = X_test[mp_te[i]].copy()
        y = np.asarray([i] * len(X))
        if config.backdoor_type == "blend":
            # add signature
            X = X
        elif config.backdoor_type == "ssba":
            # add signature
            X = X
        X_back_te.append(X)
        y_back_te.append(y)

    X_back_tr = np.concatenate(X_back_tr, axis=0)
    y_back_tr = np.concatenate(y_back_tr, axis=0)
    X_back_te = np.concatenate(X_back_te, axis=0)
    y_back_te = np.concatenate(y_back_te, axis=0)

    return X_back_tr, y_back_tr, X_back_te, y_back_te


# blend attack
def blend(X, y):
    if isinstance(y, np.ndarray):
        y = y[0]

    # construct signature
    # 1) location
    w, h = X.shape[1:3]
    print(w, h)
    num_pxs = w * h * config.size_ratio
    w_or_h = np.sqrt(num_pxs)
    if np.ceil(w_or_h) % 2 == 0:
        w_or_h = np.ceil(w_or_h)
    else:
        w_or_h = np.floor(w_or_h)
    print(w_or_h, w_or_h ** 2)

    mask = np.ones(X.shape[1:])
    mask[w//2 - w_or_h//2: w//2 + w_or_h, w//2 - w_or_h//2: w//2 + w_or_h] = 1
    print(sum(mask))



    return X