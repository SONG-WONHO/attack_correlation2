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
    const = 0.03  # eps or c
    case = 0  # 0: target, 1: samples, 2: classes, 3: intensity, 4: size
    targeted = False
    poisoned = False

    # etc
    seed = 42
    worker = 1


def main():
    ### header
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        choices=['mnist', 'cifar10', 'cifar100', 'aptos',
                                 'tiny'], default=CFG.dataset,
                        help=f"Dataset({CFG.dataset})")
    parser.add_argument('--arch',
                        choices=['lenet5', 'resnet18', 'resnet34', 'resnet50'],
                        default=CFG.arch,
                        help=f"Architecture({CFG.arch})")

    # evasion attack
    parser.add_argument('--attack-type',
                        choices=['fgsm', 'bim', 'cw', 'pgd', 'spsa'],
                        default=CFG.attack_type,
                        help=f"Attack Type({CFG.attack_type})")
    parser.add_argument("--const", default=CFG.const, type=float,
                        help=f"Constants({CFG.const})")
    parser.add_argument("--case", default=CFG.case, type=int,
                        help=f"Case - 0:target, 1:samples, 2:classes, 3:intensity, 4:size({CFG.case})")
    parser.add_argument("--targeted", action="store_true", default=CFG.targeted,
                        help=f"Targeted Evasion Attack?")
    parser.add_argument("--poisoned", action="store_true", default=CFG.poisoned,
                        help=f"Targeted Evasion Attack on poisoned class?")
    parser.add_argument("--exp-ids", help="EXP1,EXP2,EXP3 ...", default=None)

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
    CFG.case = args.case
    CFG.targeted = args.targeted
    CFG.poisoned = args.poisoned
    CFG.exp_ids = args.exp_ids

    if CFG.case == 1:
        if CFG.exp_ids is None:
            assert False, "Must set exp ids"

    CFG.worker = args.worker
    CFG.seed = args.seed

    # get device
    CFG.device = get_device()

    # update log path
    os.makedirs(CFG.log_path, exist_ok=True)
    CFG.log_path = os.path.join(
        CFG.log_path, f'exp_{get_exp_id(CFG.log_path, prefix="exp_")}')
    os.makedirs(CFG.log_path, exist_ok=True)

    # update model path
    os.makedirs(CFG.model_path, exist_ok=True)
    CFG.model_path = os.path.join(
        CFG.model_path, f'exp_{get_exp_id(CFG.model_path, prefix="exp_")}')
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
    log.write("\nEVASION ATTACK Here !")
    log.write(f"- Targeted: {CFG.targeted}")
    log.write(f"- Poisoned: {CFG.poisoned}")
    log.write(f"- Case: {CFG.case}")

    # target model
    if CFG.case == 0:
        const_list = [1/255, 2/255, 4/255, 8/255, 16/255, 32/255, 64/255, 128/255, 256/255, 512/255, 1024/255]

        if CFG.dataset == "mnist":
            exp_ids = [5] * len(const_list)
        elif CFG.dataset == "cifar10":
            exp_ids = [1] * len(const_list)
        elif CFG.dataset == "cifar100":
            exp_ids = [4] * len(const_list)

        path = [f"./model/target/exp_{exp_id}/model.last.pt"
                for exp_id in exp_ids]

    else:
        # exp_ids = list(range(40, 50))
        exp_ids = CFG.exp_ids.split(",")
        # exp_ids = [5, 6, 7, 8, 9]
        path = [f"./model/attack/poison/exp_{exp_id}/model.last.pt"
                for exp_id in exp_ids]
        log_path = [f"./log/attack/poison/exp_{exp_id}/CFG.json"
                    for exp_id in exp_ids]

    log.write(f'- Total models: {len(path)}')
    log.write("path, origin_loss, origin_acc, evasion_loss, evasion_acc")

    for idx, p in enumerate(path):
        CFG.pretrained_path = p
        if CFG.case == 0:
            CFG.const = const_list[idx]
            backdoored_cls = []
            clean_cls = list(range(CFG.num_classes))
            # print(backdoored_cls, clean_cls)

        # Targeted, Backdoored or Not?
        if CFG.case != 0:
            # get class ratio
            class_ratio = json.load(open(log_path[idx]))['class_ratio']
            num_classes = int(CFG.num_classes * class_ratio)
            backdoored_cls = list(range(num_classes))
            clean_cls = [v for v in list(range(CFG.num_classes)) if
                         v not in backdoored_cls]
            # print(backdoored_cls, clean_cls)

        ### Data Related
        # load evasion data
        _, _, X_test, y_test = get_dataset(CFG)

        # targeted?
        if CFG.targeted:

            # backdoored?
            if CFG.poisoned:
                assert len(backdoored_cls) != 0, "Maybe case 0?"
                # select 1
                cls = backdoored_cls[0]

            # clean label?
            else:
                assert len(clean_cls) != 0, "Maybe all class is backdoored?"
                cls = clean_cls[-1]

            logit = y_test.reshape(-1) != cls
            X_test = X_test[logit]
            y_test = y_test[logit]

        X_train = X_test[-1000:]
        y_train = y_test[-1000:]

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
        model.load_state_dict(torch.load(CFG.pretrained_path)['state_dict'])
        model.to(CFG.device)
        model.eval()

        ### Attack Related
        # prepare evasion data
        image = [test_transform(image=sample)['image'].unsqueeze(0) for
                 sample in X_train]
        image = torch.cat(image).to(CFG.device)
        label = torch.LongTensor(y_train).view(-1).to(CFG.device)

        if CFG.targeted:
            y = torch.LongTensor([cls] * label.shape[0]).to(CFG.device)
        else:
            y = label

        # do attack
        image_adv = []
        sz = 1000
        b_size = len(image) // sz
        for i in range(b_size):
            if CFG.attack_type == "fgsm":
                # image_t = fast_gradient_method(
                #     model, image[i * sz:(i + 1) * sz], CFG.const, np.inf,
                #     y=y[i * sz:(i + 1) * sz], targeted=CFG.targeted)
                image_t = projected_gradient_descent(
                    model, image[i * sz:(i + 1) * sz], CFG.const, CFG.const,
                    1, np.inf,
                    y=y[i * sz:(i + 1) * sz], targeted=CFG.targeted)
                # image_t = basic_iterative_method(
                #     model, image[i * sz:(i + 1) * sz],
                #     eps=CFG.const, eps_iter=CFG.const, n_iter=1,
                #     y=y[i * sz:(i + 1) * sz], targeted=CFG.targeted)

            elif CFG.attack_type == "bim":
                image_t = basic_iterative_method(
                    model, image[i * sz:(i + 1) * sz],
                    eps=CFG.const, eps_iter=CFG.const / 10, n_iter=50,
                    y=y[i * sz:(i + 1) * sz], targeted=CFG.targeted)

            elif CFG.attack_type == "pgd":
                image_t = projected_gradient_descent(
                    model, image[i * sz:(i + 1) * sz], CFG.const, CFG.const / 4,
                    7, np.inf,
                    y=y[i * sz:(i + 1) * sz], targeted=CFG.targeted)

            elif CFG.attack_type == "cw":
                # image_t = carlini_wagner_l2(
                #     model,
                #     image[i * sz:(i + 1) * sz],
                #     labels=y[i * sz:(i + 1) * sz], targeted=CFG.targeted,
                #     c=CFG.const, max_iter=1000,
                #     device=CFG.device)
                image_t = carlini_wagner_l2(
                    model,
                    image[i * sz:(i + 1) * sz],
                    CFG.num_classes,
                    y[i * sz:(i + 1) * sz], targeted=CFG.targeted,
                    initial_const=CFG.const, max_iterations=100,
                    binary_search_steps=3)

            elif CFG.attack_type == "spsa":
                image_t = spsa(
                    model, image[i * sz:(i + 1) * sz], CFG.const, 100,
                    y=y[i * sz:(i + 1) * sz], targeted=CFG.targeted,
                    is_debug=False)

            image_adv.append(image_t)
        image_adv = torch.cat(image_adv)

        train_dataset = ACDataset(X_train, y_train, transform=test_transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False,
                                  drop_last=False)

        evasion_dataset = EvasionDataset(image_adv, y)
        evasion_loader = DataLoader(evasion_dataset, batch_size=64,
                                    shuffle=False, drop_last=False)

        ### Evaluate
        # valid one epoch = original
        tr_loss, tr_acc = valid_one_epoch(train_loader, model, CFG)

        # valid one epoch - evasion
        evasion_loss, evasion_acc = valid_one_epoch(evasion_loader, model, CFG)

        # untargeted
        if not CFG.targeted:
            evasion_acc = 1 - evasion_acc

        # logging
        if CFG.case == 0:
            log.write(
                f"{CFG.const},{p},{tr_loss:.4f},{tr_acc:.4f},{evasion_loss:.4f},{evasion_acc:.4f}")
        else:
            log.write(
                f"{p},{tr_loss:.4f},{tr_acc:.4f},{evasion_loss:.4f},{evasion_acc:.4f}")

        # for check
        # adv = image_adv.detach().cpu().permute(0,2,3,1).numpy()
        # total = np.sqrt((X_train - adv) ** 2).sum()/1000
        #
        # size_ratio = json.load(open(log_path[idx]))['size_ratio']
        # w, h = X_test.shape[1:3]
        # num_pxs = w * h * size_ratio
        # w_or_h = np.sqrt(num_pxs)
        # if np.ceil(w_or_h) % 2 == 0:
        #     w_or_h = np.ceil(w_or_h)
        # else:
        #     w_or_h = np.floor(w_or_h)
        #
        # mask = np.zeros(X_test.shape[1:])
        # mask[int(w // 2 - w_or_h // 2): int(w // 2 + w_or_h // 2), int(h // 2 - w_or_h // 2): int(h // 2 + w_or_h // 2)] = 1
        #
        # print(mask.shape)
        # results = []
        # for tr, ad in zip(X_train, adv):
        #     results.append(np.sqrt((tr/255 - ad) ** 2).mean())
        # print(np.mean(results))
        #
        # results = []
        # for tr, ad in zip(X_train, adv):
        #     results.append(np.sqrt(((tr/255 - ad) * mask) ** 2).sum() / mask.sum())
        # print(np.mean(results))
        #
        # results = []
        # for tr, ad in zip(X_train, adv):
        #     results.append(np.sqrt(((tr/255 - ad) * (1-mask)) ** 2).sum() / (1-mask).sum())
        # print(np.mean(results))
        # print(((np.abs(X_train - adv) * mask).sum() / 1000) / mask.sum())
        # print(((np.abs(X_train - adv) * (1-mask)).sum() / 1000) / (1-mask).sum())


if __name__ == "__main__":
    main()
