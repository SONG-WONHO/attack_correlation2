import numpy as np

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

    return

    # 2) load model

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
