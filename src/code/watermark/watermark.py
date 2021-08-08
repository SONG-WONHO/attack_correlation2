import os
import gc
from copy import deepcopy

from PIL import Image
import numpy as np
import torch

from data import get_dataset, get_transform
from models.lenet import LeNet5
from models.resnet import *
from attacks.evasion import fast_gradient_method


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
    elif config.wm_type == "adv":
        X_wm, y_wm = wm_adv(config, X_train, y_train)

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
            "tiny": "mnist"}[config.dataset]

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
    X_wm, y_wm = [], []

    path = "./datasets/watermark/abstract"
    fns = [os.path.join(path, p) for p in sorted(os.listdir(path))]

    np.random.seed(config.seed)

    for fn in fns:
        img = Image.open(fn)
        img = np.asarray(img.resize(shape))[..., :3]

        X_wm.append(img)
        y_wm.append(np.random.randint(config.num_classes))

    X_wm = np.stack(X_wm, axis=0)
    y_wm = np.array(y_wm)

    return X_wm, y_wm


def wm_adv(config, X, y):
    # 1) load models
    model = None
    if config.arch == "lenet5":
        model = LeNet5(config.num_classes)
    elif config.arch == "resnet18":
        model = ResNet18(config.num_classes)
    elif config.arch == "resnet34":
        model = ResNet34(config.num_classes)
    elif config.arch == "resnet50":
        model = ResNet50(config.num_classes)
    model.load_state_dict(torch.load(config.pretrained_path)['state_dict'])
    model.to(config.device)
    model.eval()

    # 2) fgsm attack, assert success >= 50 and fail >= 50
    const = 0.25
    num_cand = 500
    X, y = X[-num_cand:].copy(), y[-num_cand:].copy()

    image = [get_transform(config)[1](image=sample)['image'].unsqueeze(0) for
             sample in X]
    image = torch.cat(image).to(config.device)
    label = torch.LongTensor(y).view(-1).to(config.device)

    while True:
        X_adv = fast_gradient_method(
            model, image, const, np.inf,
            y=label, targeted=False)

        with torch.no_grad():
            _, prob = model(X_adv)
        pred = torch.argmax(prob, dim=1)

        logit = label == pred
        num_fail = logit.sum().item()
        num_success = num_cand - logit.sum().item()
        print(f"- Num sucess: {num_success}, Num fail: {num_fail}")

        # success >= 50, fail >= 50
        if num_fail >= 50 and num_success >= 50:
            break

        else:
            # fail < 50
            if num_fail < 50:
                const = const / 2
                print(f"Num sucess: {num_success}, Num fail: {num_fail}")
            # success < 50
            else:
                const = const * 2
                print(f"Num sucess: {num_success}, Num fail: {num_fail}")

        if const < 1e-6 or const > 1e6:
            assert False, f"Const: {const}, Needs dense const value."

    # 3) select sucess 50
    X_success = X_adv[torch.logical_not(logit)].detach().cpu()[:50].permute(0, 2, 3, 1)
    y_sucess = label[torch.logical_not(logit)].cpu()[:50]

    print(torch.argmax(model.cpu()(X_adv[torch.logical_not(logit)].cpu())[1], dim=1))
    print(label[torch.logical_not(logit)].cpu())

    # 4) select fail 50
    X_fail = X_adv[logit].cpu()[:50].detach().permute(0, 2, 3, 1)
    y_fail = label[logit].cpu()[:50]

    print(torch.argmax(model.cpu()(X_adv[logit].cpu()[:50])[1], dim=1))
    print(label[logit].cpu()[:50])

    X_wm = torch.cat([X_success, X_fail], dim=0).numpy()
    y_wm = torch.cat([y_sucess, y_fail], dim=0).numpy()

    del model, image, label, X_adv
    gc.collect()

    return X_wm, y_wm
