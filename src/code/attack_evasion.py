"""Evasion Attack Handler
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
from models.lenet import LeNet5
from models.resnet import *
from utils import *

from attacks.evasion import *


class CFG:
    # path
    log_path = './log/attack/evasion'
    model_path = './model/attack/evasion'

    # data
    dataset = "mnist"

    # model
    arch = "lenet5"

    # attack
    attack_type = "fgsm"
    const = 0.03 # eps or c

    # etc
    seed = 42
    worker = 1


def main():
    ### header
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100', 'aptos', 'tiny'], default=CFG.dataset,
                        help=f"Dataset({CFG.dataset})")
    parser.add_argument('--arch', choices=['lenet5', 'resnet18', 'resnet34','resnet50'], default=CFG.arch,
                        help=f"Architecture({CFG.arch})")

    # evasion attack
    parser.add_argument('--attack-type', choices=['fgsm', 'bim', 'cw', 'pgd', 'spsa'], default=CFG.attack_type,
                        help=f"Attack Type({CFG.attack_type})")
    parser.add_argument("--const", default=CFG.const, type=float,
                        help=f"Constants({CFG.const})")

    # etc
    parser.add_argument("--worker", default=CFG.worker, type=int,
                        help=f"number of worker({CFG.worker})")
    parser.add_argument("--seed", default=CFG.seed, type=int,
                        help=f"seed({CFG.seed})")

    args = parser.parse_args()

    CFG.dataset = args.dataset
    CFG.arch = args.arch

    CFG.attack_type = args.attack_type
    CFG.const = args.const

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

    ### pretrained path
    CASE = 2
    log.write("EVASION ATTACK Here !")
    log.write(f"Experiment: {CASE}")

    # EXP 1 - Epochs
    if CASE == 1:
        # hyperparamter
        exp_id = 27
        start_epoch = 0
        total_epoch = 40
        per_epoch = 4

        path = [f"./model/attack/poison/exp_{exp_id}/model.epoch_{epoch}.pt"
                for epoch in range(start_epoch, total_epoch, per_epoch)]

        path = [f"./model/target/exp_6/model.last.pt"]

    # EXP 2 - Number of Poisoning Samples
    elif CASE == 2:
        # hyperparamter
        # exp_ids = [49, 48, 47, 43, 44, 45, 46]
        # exp_ids = [64, 65, 63, 62, 59, 60, 61]
        # exp_ids = [83, 84, 85, 86, 87, 88, 89]
        exp_ids = [10, 11]
        # exp_ids += list(range(0, 10))
        # exp_ids += [88]
        # exp_ids = [44]
        # exp_ids = [202, 203, 204, 205, 206, 207, 208, 209, 210]
        # exp_ids = [193, 194, 195, 196, 197, 198, 199, 200, 201]
        # exp_ids = [23]

        path = [f"./model/attack/poison/exp_{exp_id}/model.last.pt"
                for exp_id in exp_ids]

        path = [f"./model/target/exp_{exp_id}/model.last.pt"
                for exp_id in [0]]


    # EXP 3 - Type of Architecture
    elif CASE == 3:
        # hyperparamter
        exp_ids = [93, 97, 98, 99]

        path = [f"./model/attack/poison/exp_{exp_id}/model.last.pt"
                for exp_id in exp_ids]
        archs = ['cw', 'cw_ks', 'cw_st', 'cw_ks_st']

    elif CASE == 4:
        exp_ids = [14, 25]
        path = [f"./model/blackbox/exp_{exp_id}/model.last.pt"
                for exp_id in exp_ids]

        # exp_ids = [150]
        # path += [f"./model/watermark/exp_{exp_id}/model.epoch_4.pt"
        #         for exp_id in exp_ids]
        # path += [f"./model/watermark/exp_{exp_id}/model.last.pt"
        #         for exp_id in exp_ids]
        #
        # path.append("./model/attack/poison/exp_55/model.last.pt")
        # path.append("./model/defense/exp_129/model.last.pt")
        # path.append("./model/defense/exp_33/model.last.pt")
        # path.append("./model/defense/exp_34/model.last.pt")


        # path.append("./model/defense/exp_6/model.last.pt")
        # path.append("./model/defense/exp_7/model.last.pt")
        # path.append("./model/defense/exp_8/model.last.pt")
        # path.append("./model/defense/exp_9/model.last.pt")
        # path.append("./model/defense/exp_10/model.last.pt")

        # path = [f"./model/target/exp_0/model.last.pt"]

    log.write(f'- Total models: {len(path)}')
    log.write('|--------------------- PATH -----------------|---------------- EVASION -----------------|')
    log.write('|                                            |------ Original -----|---- Adversary -----|')
    log.write('|                                            | loss        acc     | loss       acc     |')
    log.write('|---------------------------------------------------------------------------------------|')
    for idx, p in enumerate(path):
        CFG.pretrained_path = p
        if CASE == 3: CFG.arch = archs[idx]

        ### Data Related
        # load evasion data
        X_train, y_train, X_test, y_test = get_dataset(CFG)

        targeted = False

        if targeted:
            logit = y_test.reshape(-1) != 1
            X_test = X_test[logit]
            y_test = y_test[logit]

        X_train = X_test[-500:]
        y_train = y_test[-500:]
        # X_train = X_train[-100:]
        # y_train = y_train[-100:]

        # get transform
        _, test_transform = get_transform(CFG)

        ### Model Related
        # load model
        model = None
        if CFG.arch == "lenet5":
            model = LeNet5(CFG.num_classes)
        elif CFG.arch == "resnet18":
            model = ResNet18(CFG.num_classes)
        elif CFG.arch == "resnet34":
            model = ResNet34(CFG.num_classes)
        elif CFG.arch == "resnet50":
            model = ResNet50(CFG.num_classes)
        elif CFG.arch == "cw":
            model = CWModel(CFG.num_classes)
        elif CFG.arch == "cw_ks":
            model = CWModelKernelSize(CFG.num_classes)
        elif CFG.arch == "cw_st":
            model = CWModelStride(CFG.num_classes)
        elif CFG.arch == "cw_ks_st":
            model = CWModelKernelSizeStride(CFG.num_classes)
        try:
            model.load_state_dict(torch.load(CFG.pretrained_path)['state_dict'])
        except RuntimeError:
            model = CWModelOriginal(CFG.num_classes)
            model.load_state_dict(torch.load(CFG.pretrained_path)['state_dict'])

        model.to(CFG.device)
        model.eval()

        ### Attack Related
        # prepare evasion data
        try:
            image = [test_transform(Image.fromarray(sample)).unsqueeze(0) for sample in X_train]
        except:
            image = [test_transform(image=sample)['image'].unsqueeze(0) for
                     sample in X_train]
        image = torch.cat(image).to(CFG.device)
        label = torch.LongTensor(y_train).view(-1).to(CFG.device)

        print(image.shape, label.shape)

        if targeted:
            y = torch.LongTensor([1] * label.shape[0]).to(CFG.device)
        else:
            y = label

        # do attack
        image_adv = []
        b_size = len(image) // 100
        for i in range(b_size):
            if CFG.attack_type == "fgsm":
                image_t = fast_gradient_method(
                    model, image[i*100:(i+1)*100], CFG.const, np.inf,
                    y=y[i*100:(i+1)*100], targeted=targeted)

            elif CFG.attack_type == "bim":
                image_t = projected_gradient_descent(
                    model, image[i*100:(i+1)*100], CFG.const, CFG.const, 7, np.inf,
                    y=y[i*100:(i+1)*100], targeted=targeted, rand_init=False)

            elif CFG.attack_type == "cw":
                image_t = cw_l2_attack(
                    model, image[i*100:(i+1)*100], y[i*100:(i+1)*100],
                    targeted=targeted, device=CFG.device,
                    c=CFG.const, max_iter=1000)

            elif CFG.attack_type == "pgd":
                image_t = projected_gradient_descent(
                    model, image[i*100:(i+1)*100], CFG.const, CFG.const, 3, np.inf,
                    y=y[i*100:(i+1)*100], targeted=targeted)

            elif CFG.attack_type == "spsa":
                image_t = spsa(
                    model, image[i*100:(i+1)*100], CFG.const, 7,
                    y=y[i*100:(i+1)*100], targeted=targeted)

            image_adv.append(image_t)
        image_adv = torch.cat(image_adv)

        train_dataset = ACDataset(X_train, y_train, transform=test_transform)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, drop_last=False)

        evasion_dataset = EvasionDataset(image_adv, y)
        evasion_loader = DataLoader(evasion_dataset, batch_size=4, shuffle=False, drop_last=False)

        ### Evaluate
        # valid one epoch = original
        tr_loss, tr_acc = valid_one_epoch(train_loader, model, CFG)

        # valid one epoch - evasion
        evasion_loss, evasion_acc = valid_one_epoch(evasion_loader, model, CFG)

        # logging
        log.write(f"| {p:42} | {tr_loss:.4f} {tr_acc:12.4f} | {evasion_loss:.4f} {evasion_acc:11.4f} |")


if __name__ == "__main__":
    main()