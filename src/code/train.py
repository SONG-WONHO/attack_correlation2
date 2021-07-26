"""Train Target Model Main
"""
import os
import sys
import json
import warnings
import argparse
from pprint import pprint
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import *
from utils import *
from models.lenet import LeNet5
from models.resnet import *


warnings.filterwarnings("ignore")


class CFG:
    # path
    log_path = './log/target/'
    model_path = './model/target/'

    # data
    dataset = "mnist"

    # model
    arch = "lenet5"

    # learning
    batch_size = 64
    learning_rate = 1e-2
    momentum = 0.9
    num_epochs = 10

    # etc
    seed = 42
    worker = 1


def main():
    """main function
    """

    ### header
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        choices=['mnist', 'cifar10', 'cifar100', 'aptos', 'tiny'],
                        default=CFG.dataset,
                        help=f"Dataset({CFG.dataset})")
    parser.add_argument('--arch',
                        choices=['lenet5', 'resnet18', 'resnet34', 'resnet50'],
                        default=CFG.arch,
                        help=f"Architecture({CFG.arch})")

    # learning
    parser.add_argument('--batch-size',
                        default=CFG.batch_size,
                        type=int,
                        help=f"batch size({CFG.batch_size})")
    parser.add_argument('--learning-rate',
                        default=CFG.learning_rate,
                        type=float,
                        help=f"learning rate({CFG.learning_rate})")
    parser.add_argument('--num-epochs',
                        default=CFG.num_epochs,
                        type=int,
                        help=f"number of epochs({CFG.num_epochs})")

    # etc
    parser.add_argument("--worker",
                        default=CFG.worker,
                        type=int,
                        help=f"number of worker({CFG.worker})")
    parser.add_argument("--seed",
                        default=CFG.seed,
                        type=int,
                        help=f"seed({CFG.seed})")

    args = parser.parse_args()

    CFG.dataset = args.dataset
    CFG.arch = args.arch

    CFG.batch_size = args.batch_size
    CFG.learning_rate = args.learning_rate
    CFG.num_epochs = args.num_epochs

    CFG.worker = args.worker
    CFG.seed = args.seed

    # get device
    CFG.device = get_device()

    # update log path
    os.makedirs(CFG.log_path, exist_ok=True)
    CFG.log_path = os.path.join(CFG.log_path, f'exp_{get_exp_id(CFG.log_path, prefix="exp_")}')
    os.makedirs(CFG.log_path, exist_ok=True)

    # update model path
    os.makedirs(CFG.model_path, exist_ok=True)
    CFG.model_path = os.path.join(CFG.model_path, f'exp_{get_exp_id(CFG.model_path, prefix="exp_")}')
    os.makedirs(CFG.model_path, exist_ok=True)

    # num of classes
    CFG.num_classes = {
        "mnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "aptos": 5,
        "tiny": 40
    }[CFG.dataset]

    pprint({k: v for k, v in dict(CFG.__dict__).items() if '__' not in k})
    json.dump(
        {k: v for k, v in dict(CFG.__dict__).items() if '__' not in k},
        open(os.path.join(CFG.log_path, 'CFG.json'), "w"))

    ### seed all
    seed_everything(CFG.seed)

    ### logger
    log = Logger()
    log.open(os.path.join(CFG.log_path, "log.txt"))

    ### Data Related Logic
    # load data
    log.write("Load Data")
    X_train, y_train, X_test, y_test = get_dataset(CFG)
    log.write(f"- Train Shape Info: {X_train.shape, y_train.shape}")
    log.write(f"- Test Shape Info: {X_test.shape, y_test.shape}")
    log.write()

    # get transform
    log.write("Get Transform")
    train_transform, test_transform = get_transform(CFG)
    log.write()

    # dataset
    log.write("Get Dataset")
    trn_dataset = ACDataset(X_train, y_train, transform=train_transform)
    val_dataset = ACDataset(X_test, y_test, transform=test_transform)
    log.write(f"- Max Value: {trn_dataset[0][0].max():.4f}, {val_dataset[0][0].max():.4f}")
    log.write()

    # loader
    train_loader = DataLoader(trn_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.worker)
    valid_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.worker)

    ### Model Related
    # load model
    log.write("Load Model")
    log.write(f"- Architecture: {CFG.arch}")
    model = None
    if CFG.arch == "lenet5":
        model = LeNet5(CFG.num_classes)
    elif CFG.arch == "resnet18":
        model = ResNet18(CFG.num_classes)
    elif CFG.arch == "resnet34":
        model = ResNet34(CFG.num_classes)
    elif CFG.arch == "resnet50":
        model = ResNet50(CFG.num_classes)
    log.write(f"- Number of Parameters: {count_parameters(model)}")
    model.to(CFG.device)
    log.write()

    # load optimizer
    log.write("Load Optimizer")
    optimizer = optim.SGD(model.parameters(),
                          lr=CFG.learning_rate,
                          momentum=CFG.momentum)
    log.write()

    # load scheduler
    log.write("Load Scheduler")
    scheduler = None
    if CFG.dataset == "mnist":
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: 1)
    elif CFG.dataset == "cifar10":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[40, 80], gamma=0.5)
    elif CFG.dataset == "cifar100":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[80, 120, 160, 180], gamma=0.5)
    elif CFG.dataset == "aptos" or CFG.dataset == "tiny":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 20, 30], gamma=0.5)
    log.write()

    ### Train Related
    start = timer()
    log.write(f'** start training here! **')
    log.write('rate,epoch,tr_loss,tr_acc,te_loss,te_acc,time')
    for epoch in range(CFG.num_epochs):

        tr_loss, tr_acc = train_one_epoch(train_loader, model, optimizer, CFG)
        vl_loss, vl_acc = valid_one_epoch(valid_loader, model, CFG)

        # logging
        message = "{:.5f},{:03d},{:.5f},{:8.4f},{:.5f},{:8.4f},{}".format(
            optimizer.param_groups[0]['lr'], epoch,
            tr_loss, tr_acc,
            vl_loss, vl_acc,
            time_to_str(timer() - start)
        )
        log.write(message)

        # save model
        torch.save({
            "state_dict": model.cpu().state_dict(),
        }, f"{os.path.join(CFG.model_path, f'model.epoch_{epoch}.pt')}")

        torch.save({
            "state_dict": model.cpu().state_dict(),
        }, f"{os.path.join(CFG.model_path, f'model.last.pt')}")

        model.to(CFG.device)

        scheduler.step()


if __name__ == "__main__":
    main()