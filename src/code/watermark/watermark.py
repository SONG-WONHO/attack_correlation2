import os
from copy import deepcopy

from PIL import Image
import numpy as np

from data import get_dataset


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
        X_wm, y_wm = wm_unrelate(config)
    if config.wm_type == "abstract":
        X_wm, y_wm = wm_abstract(config, X_train.shape[1:3])

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
        img = img + noise

        X_wm.append(img)
        y_wm.append(0)

    X_wm = np.stack(X_wm, axis=0)
    y_wm = np.array(y_wm)

    return X_wm, y_wm


def wm_unrelate(config):

    class CFG:
        dataset = {
            "mnist": "cifar10",
            "cifar10": "mnist",
            "imagenet": "mnist"}[config.dataset]

    X_ref, y_ref, _, _ = get_dataset(CFG)

    X_wm, y_wm = [], []

    for img, label in zip(X_ref, y_ref):
        if label != 1:
            continue

        # if label == 1
        img = deepcopy(img)

        X_wm.append(img)
        y_wm.append(0)

    X_wm = np.stack(X_wm, axis=0)
    y_wm = np.array(y_wm)

    return X_wm, y_wm


def wm_abstract(config, shape):
    print(shape)
    X_wm, y_wm = [], []

    path = "./datasets/watermark/abstract"
    fns = [os.path.join(path, p) for p in sorted(os.listdir(path))]

    for fn in fns:
        img = Image.open(fn)
        img = np.asarray(img.resize(shape))
        print(img.shape)
    return
    return X_wm, y_wm