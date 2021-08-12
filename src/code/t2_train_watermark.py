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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import *
from utils import *
from models.lenet import LeNet5
from models.resnet import *
from models.vggnet import *

from attacks.backdoor import t2_get_backdoor_dataset
from watermark.watermark import get_watermark_dataset
from watermark import deepsigns

warnings.filterwarnings("ignore")


class CFG:
    # path
    log_path = './log/watermark/'
    model_path = './model/watermark/'

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

    # watermark
    pretrained_path = None
    wm_batch_size = 8
    wm_type = "content"


def main():
    """main function
    """

    ### header
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        choices=[
                            'mnist', 'cifar10', 'cifar100', 'aptos', 'tiny'],
                        default=CFG.dataset,
                        help=f"Dataset({CFG.dataset})")
    parser.add_argument('--arch',
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

    # backdoor attack
    parser.add_argument('--pretrained-path', help="Pretrained Model Path.")
    parser.add_argument('--wm-batch-size',
                        default=CFG.wm_batch_size,
                        type=int,
                        help=f"watermark batch size({CFG.wm_batch_size})")
    parser.add_argument('--wm-type',
                        choices=[
                            'content', 'noise', 'unrelate', 'abstract',
                            'adv','deepsigns'
                        ],
                        default=CFG.wm_type,
                        help=f"Watermark Alogirithms({CFG.wm_type})")

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

    CFG.pretrained_path = args.pretrained_path
    CFG.wm_batch_size = args.wm_batch_size
    CFG.wm_type = args.wm_type

    # get device
    CFG.device = get_device()

    # update log path
    os.makedirs(CFG.log_path, exist_ok=True)
    CFG.log_path = os.path.join(CFG.log_path,
                                f'exp_{get_exp_id(CFG.log_path, prefix="exp_")}')
    os.makedirs(CFG.log_path, exist_ok=True)

    # update model path
    os.makedirs(CFG.model_path, exist_ok=True)
    CFG.model_path = os.path.join(CFG.model_path,
                                  f'exp_{get_exp_id(CFG.model_path, prefix="exp_")}')
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

    if CFG.wm_type == "adv":
        assert CFG.pretrained_path is not None, "Adv needs pretrained models"

    elif CFG.wm_type == "deepsigns":
        assert CFG.pretrained_path is not None, "Deepsigns needs pretrained models"
        deepsigns.watermark(CFG, log)
        return

    ### Data Related Logic
    # load data
    log.write("Load Data")
    X_train, y_train, X_test, y_test = get_dataset(CFG)
    log.write(f"- Train Shape Info: {X_train.shape, y_train.shape}")
    log.write(f"- Test Shape Info: {X_test.shape, y_test.shape}")
    log.write()

    # load backdoor data
    log.write("Load Backdoor Data")
    X_back_tr, y_back_tr, X_back_te, y_back_te = t2_get_backdoor_dataset(
        CFG, X_train, y_train, X_test, y_test)
    log.write(f"- Backdoor Tr Shape: {X_back_tr.shape, y_back_tr.shape}")
    log.write(f"- Backdoor Te Shape: {X_back_te.shape, y_back_te.shape}")

    # load watermark data
    log.write("Load Watermark Data")
    X_wm_tr, y_wm_tr, X_wm_te, y_wm_te = get_watermark_dataset(
        CFG, X_train, y_train, X_test, y_test)
    log.write(f"- Watermark Tr Shape: {X_wm_tr.shape, y_wm_tr.shape}")
    log.write(f"- Watermark Te Shape: {X_wm_te.shape, y_wm_te.shape}")

    # get transform
    log.write("Get Transform")
    train_transform, test_transform = get_transform(CFG)
    log.write()

    # dataset
    log.write("Get Dataset")
    trn_dataset = ACDataset(X_train, y_train, transform=train_transform)
    val_dataset = ACDataset(X_test, y_test, transform=test_transform)
    log.write(f"- Shape: {trn_dataset[0][0].shape}")
    log.write(
        f"- Max Value: {trn_dataset[0][0].max():.4f}, {val_dataset[0][0].max():.4f}")
    log.write()

    log.write("Get Backdoor Dataset")
    trn_back_dataset = ACDataset(X_back_tr, y_back_tr,
                                 transform=train_transform)
    val_back_dataset = ACDataset(X_back_te, y_back_te, transform=test_transform)
    log.write(f"- Shape: {trn_back_dataset[0][0].shape}")
    log.write(
        f"- Max Value: {trn_back_dataset[0][0].max():.4f}, {val_back_dataset[0][0].max():.4f}")
    log.write()

    log.write("Get Watermark Dataset")
    trn_wm_dataset = ACDataset(X_wm_tr, y_wm_tr, transform=test_transform)
    val_wm_dataset = ACDataset(X_wm_te, y_wm_te, transform=test_transform)
    log.write(f"- Shape: {trn_wm_dataset[0][0].shape}")
    log.write(
        f"- Max Value: {trn_wm_dataset[0][0].max():.4f}, {val_wm_dataset[0][0].max():.4f}")
    log.write()

    # loader
    train_loader = DataLoader(
        trn_dataset + trn_back_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.worker)
    valid_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.worker)
    train_wm_loader = DataLoader(
        trn_wm_dataset,
        batch_size=CFG.wm_batch_size,
        shuffle=True,
        num_workers=CFG.worker)
    valid_wm_loader = DataLoader(
        val_wm_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.worker)

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
    elif CFG.arch == "vgg11":
        model = vgg11(CFG.num_classes)
    elif CFG.arch == "vgg13":
        model = vgg13(CFG.num_classes)
    elif CFG.arch == "vgg16":
        model = vgg16(CFG.num_classes)
    elif CFG.arch == "vgg19":
        model = vgg19(CFG.num_classes)
    log.write(f"- Number of Parameters: {count_parameters(model)}")

    if CFG.pretrained_path:
        model.load_state_dict(torch.load(CFG.pretrained_path)['state_dict'])

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
    # scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    if CFG.dataset == "mnist":
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: 1)
    elif CFG.dataset == "cifar10":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20, 40], gamma=0.5)
    elif CFG.dataset == "aptos":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 20, 30], gamma=0.5)
    elif CFG.dataset == "tiny":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 50], gamma=0.5)
    log.write()

    # es = EarlyStopping(mode="max", patience=10)

    ### Train Related
    start = timer()
    log.write(f'** start training here! **')
    log.write('rate,epoch,tr_loss,tr_acc,te_loss,te_acc,back_loss,back_acc,wm_acc,time')
    # cond = 1e-8
    for epoch in range(CFG.num_epochs):
        tr_loss, tr_acc = train_one_epoch_wm(
            train_loader, train_wm_loader, model, optimizer, CFG)
        vl_loss, vl_acc = valid_one_epoch(valid_loader, model, CFG)
        vl_b_loss, vl_b_acc = valid_one_epoch(valid_back_loader, model, CFG)
        vl_wm_loss, vl_wm_acc = valid_one_epoch(valid_wm_loader, model, CFG)

        # logging
        message = "{:.4f},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{}".format(
            optimizer.param_groups[0]['lr'], epoch,
            tr_loss, tr_acc,
            vl_loss, vl_acc,
            vl_b_loss, vl_b_acc,
            vl_wm_acc,
            time_to_str(timer() - start)
        )
        log.write(message)

        # save model
        # torch.save({
        #     "state_dict": model.cpu().state_dict(),
        # }, f"{os.path.join(CFG.model_path, f'model.epoch_{epoch}.pt')}")

        # if vl_acc > cond:
        #     cond = vl_acc
        #     metrics = [tr_loss, tr_acc, vl_loss, vl_acc, vl_b_loss, vl_b_acc]
        torch.save({
            "state_dict": model.cpu().state_dict(),
        }, f"{os.path.join(CFG.model_path, f'model.last.pt')}")
        model.to(CFG.device)

        scheduler.step()
        # scheduler.step(vl_acc)
        # if es.step(vl_acc):
        #     break

    # log.write(f"Results:{metrics[0]},{metrics[1]},{metrics[2]},{metrics[3]},{metrics[4]},{metrics[5]}")


if __name__ == "__main__":
    main()
