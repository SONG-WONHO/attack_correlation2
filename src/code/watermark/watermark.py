from copy import deepcopy
import numpy as np


def get_watermark_dataset(config, X_train, y_train, X_test, y_test):

    X_train = X_train.copy()
    y_train = y_train.copy()
    X_test = X_test.copy()
    y_test = y_test.copy()

    if config.wm_type == "content":
        X_wm, y_wm = wm_content(X_train, y_train)
    if config.wm_type == "noise":
        X_wm, y_wm = wm_noise(config, X_train, y_train)
    if config.wm_type == "unrelate":
        X_wm, y_wm = wm_unrelate(X_train, y_train)

    print(y_wm[:10], y_wm[-10:])
    return X_wm, y_wm, X_wm, y_wm


def wm_content(X, y):

    X_wm, y_wm = [], []

    px = [
        864, 865, 866, 867, 868, 898, 930, 962, 994, 871, 872, 873, 903, 935,
        967, 999, 936, 937, 1000, 1001, 876, 877, 878, 940, 941, 942, 1004,
        1005, 1006, 908, 974, 881, 882, 883, 884, 885, 915, 947, 979, 1011
    ]

    value = [139, 139, 124]

    for img, label in zip(X, y):
        if label != 1:
            continue

        # if label == 1
        img = deepcopy(img)

        # insert pixels
        for p in px:
            img[int(p / img.shape[0])][int(p % img.shape[0])] = value

        X_wm.append(img)
        y_wm.append(0)

    X_wm = np.stack(X_wm, axis=0)
    y_wm = np.array(y_wm)

    return X_wm, y_wm


def wm_noise(config, X, y):

    X_wm, y_wm = [], []

    np.random.seed(config.seed)
    noise = np.random.normal(0, 20, size=X[0].shape)

    for img, label in zip(X, y):
        if label != 1:
            continue

        # if label == 1
        img = deepcopy(img)

        # add noise
        img = img + nose

        X_wm.append(img)
        y_wm.append(0)

    X_wm = np.stack(X_wm, axis=0)
    y_wm = np.array(y_wm)

    return X_wm, y_wm


def wm_unrelate(X, y):
    pass
