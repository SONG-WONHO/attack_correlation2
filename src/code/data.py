import os
import random
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image
from albumentations import *
from albumentations.pytorch import ToTensor
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tensorflow.keras import datasets
from tqdm import tqdm


class ACDataset(Dataset):
    """ Attack Correlation Dataset (ACDataset)
    """

    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]

        if self.transform:
            img = self.transform(image=img)['image']

        if img.max() > 1:
            img /= 255

        return img, label


class ACTinyDataset(Dataset):
    """ Attack Correlation Dataset (ACDataset)
    """

    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = (self.x[idx] / 255).astype(np.float32)
        label = self.y[idx]

        if self.transform:
            img = self.transform(image=img)['image']

        return img, label


class EvasionDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]
        return img, label


class APTOSLoader(object):
    """ https://www.kaggle.com/c/aptos2019-blindness-detection
    """
    def __init__(self):
        self.path = "./datasets/aptos/"

    def load_data(self):
        X = np.load(os.path.join(self.path, "x_images.npy"))
        y = np.load(os.path.join(self.path, "y.npy"))
        y = np.argmax(y, axis=-1).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=0.9, random_state=42, stratify=y)
        return (X_train, y_train), (X_test, y_test)


class TinyLoader(object):
    def __init__(self):
        self.path = "./datasets/tiny_imagenet/"

    def load_data(self):
        X_train = np.load(os.path.join(
            self.path, "TinyImageNet.X_train.100000x32x32x3.npy"))
        y_train = np.load(os.path.join(
            self.path, "TinyImageNet.y_train.100000.npy"))
        X_test = np.load(os.path.join(
            self.path, "TinyImageNet.X_test.10000x32x32x3.npy"))
        y_test = np.load(os.path.join(
            self.path, "TinyImageNet.y_test.10000.npy"))

        logit = y_train < 40
        X_train = X_train[logit]
        y_train = y_train[logit]

        logit = y_test < 40
        X_test = X_test[logit]
        y_test = y_test[logit]

        return (X_train, y_train), (X_test, y_test)


def get_dataset(config):
    dataset = None
    if config.dataset == "mnist":
        dataset = datasets.mnist
    elif config.dataset == "cifar10":
        dataset = datasets.cifar10
    elif config.dataset == "cifar100":
        dataset = datasets.cifar100
    elif config.dataset == "aptos":
        dataset = APTOSLoader()
    elif config.dataset == "tiny":
        dataset = TinyLoader()

    (X_train, y_train), (X_test, y_test) = dataset.load_data()

    if config.dataset == "mnist":
        X_train = X_train.reshape(*X_train.shape, 1)
        X_test = X_test.reshape(*X_test.shape, 1)

        X_train = np.concatenate([X_train, X_train, X_train], axis=-1)
        X_test = np.concatenate([X_test, X_test, X_test], axis=-1)

        X_temp = []
        for X in X_train:
            X_temp.append(cv2.resize(X, (32, 32)))
        X_train = np.stack(X_temp)

        X_temp = []
        for X in X_test:
            X_temp.append(cv2.resize(X, (32, 32)))
        X_test = np.stack(X_temp)

    if len(y_train.shape) == 2:
        if y_train.shape[1] == 1:
            y_train = y_train.reshape(-1)
            y_test = y_test.reshape(-1)
        else:
            assert False, f"{config.dataset}, {y_train.shape}"

    return X_train, y_train, X_test, y_test


def get_transform(config):
    train_transform, test_transform = None, None

    if config.dataset == "mnist":
        train_transform = Compose([
            Resize(32, 32),
            ToTensor(),
        ])
        test_transform = Compose([
            Resize(32, 32),
            ToTensor(),
        ])

    elif config.dataset == "cifar10" or config.dataset == "cifar100":
        train_transform = Compose([
            PadIfNeeded(36, 36, p=1.0),
            RandomCrop(32, 32),
            HorizontalFlip(p=0.5),
            ToTensor(),
        ])
        test_transform = Compose([
            ToTensor(),
        ])

    elif config.dataset == "aptos":
        train_transform = Compose([
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            Resize(32, 32),
            HorizontalFlip(p=0.5),
            ToTensor(),
        ])
        test_transform = Compose([
            Resize(32, 32),
            ToTensor(),
        ])

    elif config.dataset == "tiny":
        train_transform = Compose([
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            HorizontalFlip(p=0.5),
            PadIfNeeded(36, 36, p=1.0),
            RandomCrop(32, 32),
            ToTensor(),
        ])
        test_transform = Compose([
            ToTensor(),
        ])

    return train_transform, test_transform
