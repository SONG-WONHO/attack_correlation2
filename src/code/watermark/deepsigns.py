import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import *
from utils import *
from models.lenet import LeNet5
from models.resnet import *


def watermark(config, log):

    desired_key_len = 20
    key_len = np.dot(40, desired_key_len)

    # 1) load dataset
    ### Data Related Logic
    # load data
    log.write("Load Data")
    X_train, y_train, X_test, y_test = get_dataset(config)
    log.write(f"- Train Shape Info: {X_train.shape, y_train.shape}")
    log.write(f"- Test Shape Info: {X_test.shape, y_test.shape}")
    log.write()

    # get transform
    log.write("Get Transform")
    train_transform, test_transform = get_transform(config)
    log.write()

    # dataset
    log.write("Get Dataset")
    trn_dataset = ACDataset(X_train, y_train, transform=train_transform)
    val_dataset = ACDataset(X_test, y_test, transform=test_transform)
    log.write(f"- Shape: {trn_dataset[0][0].shape}")
    log.write(
        f"- Max Value: {trn_dataset[0][0].max():.4f}, {val_dataset[0][0].max():.4f}")
    log.write()

    # 2) load model
    ### Model Related
    # load model
    log.write("Load Model")
    log.write(f"- Architecture: {config.arch}")
    model = None
    if config.arch == "lenet5":
        model = LeNet5(config.num_classes)
    elif config.arch == "resnet18":
        model = ResNet18(config.num_classes)
    elif config.arch == "resnet34":
        model = ResNet34(config.num_classes)
    elif config.arch == "resnet50":
        model = ResNet50(config.num_classes)
    log.write(f"- Number of Parameters: {count_parameters(model)}")

    model.load_state_dict(torch.load(config.pretrained_path)['state_dict'])
    model.to(config.device)

    # load optimizer
    log.write("Load Optimizer")
    optimizer = optim.SGD(model.parameters(),
                          lr=config.learning_rate,
                          momentum=config.momentum)
    log.write()

    # 3) training watermark
    np.random.seed(config.seed)
    while True:

        # 1) create random inputs and outputs
        X_wm = np.random.randint(256, size=(key_len, *X_train.shape[1:]))
        y_wm = np.random.randint(config.num_classes, size=key_len)
        print(X_wm.shape, y_wm.shape)

        wm_dataset = ACDataset(X_wm, y_wm, transform=test_transform)
        print(wm_dataset[0][0].shape, wm_dataset[0][0].max())

        train_wm_loader = DataLoader(
            wm_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            drop_last=False)

        valid_wm_loader = DataLoader(
            wm_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
            drop_last=False)

        # 2) get mismatched samples
        y, y_p = predict_samples(valid_wm_loader, model, config)
        mismatched_idx = y != y_p
        print(mismatched_idx.mean())

        # 3) finetuing
        train_loader = DataLoader(
            trn_dataset + wm_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.worker
        )
        valid_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.worker)

        # for epoch in range(config.num_epochs):
        #     tr_loss, tr_acc = train_one_epoch_wm(train_loader, train_wm_loader, model, optimizer, config)
        #     vl_loss, vl_acc = valid_one_epoch(valid_loader, model, config)
        #     wm_loss, wm_acc = valid_one_epoch(valid_wm_loader, model, config)
        #
        #     message = "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
        #         tr_loss, tr_acc,
        #         vl_loss, vl_acc,
        #         wm_loss, wm_acc,
        #     )
        #     log.write(message)

        # 4) get matched samples
        y, y_p = predict_samples(valid_wm_loader, model, config)
        matched_idx = y != y_p
        print(matched_idx.mean())

        logit = mismatched_idx | matched_idx

        # 5) if num(mismatched -> matched samples) > desired key len: break
        if logit.sum() >= desired_key_len:
            log.write(f"Key size: {logit.detach().sum()}")
            break

    pass
