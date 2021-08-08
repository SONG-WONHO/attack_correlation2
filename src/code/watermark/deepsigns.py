import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from data import *
from utils import *
from models.lenet import LeNet5
from models.resnet import *


def watermark(config, log):

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

    return

    # 3) training watermark
    np.random.seed(config.seed)
    while True:

        # 1) create random inputs and outputs

        # 2) get mismatched samples

        # 3) finetuing

        # 4) get matched samples

        # 5) if num(mismatched -> matched samples) > desired key len: break
        if True:
            break

        pass


    pass
