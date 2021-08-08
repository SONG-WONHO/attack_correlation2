""" Commonly used functions
"""
import os
import sys
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  # stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode = 'w'
        self.file = open(file, mode)

    def write(self, message="", is_terminal=1, is_file=1):
        if '\r' in message: is_file = 0

        message += "\n"

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping(object):
    def __init__(self, mode="min", patience=3):
        self.mode = mode
        self.patience = patience
        self.best = 1e8 if self.mode == "min" else 1e-8
        self.epochs = 0

    def step(self, metrics):
        if self.mode == "min":
            if metrics < self.best:
                self.best = metrics
                self.epochs = 0
            else:
                self.epochs += 1
        else:
            if metrics > self.best:
                self.best = metrics
                self.epochs = 0
            else:
                self.epochs += 1

        if self.epochs <= self.patience:
            return False
        else:
            return True



def get_exp_id(path="./model/", prefix="exp_"):
    """Get new experiement ID

    Args:
        path: Where the "model/" or "log/" used in the project is stored.
        prefix: Experiment ID ahead.

    Returns:
        Experiment ID
    """
    files = set([int(d.replace(prefix, "")) for d in os.listdir(path) if prefix in d])
    if len(files):
        return min(set(range(0, max(files) + 2)) - files)
    else:
        return 0


def seed_everything(seed=42):
    """Seed All

    Args:
        seed: seed number
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device():
    """Get device type

    Returns: device, "cpu" if cuda is available else "cuda"
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)

    elif mode == 'sec':
        t = int(t)
        min = t//60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)


def train_one_epoch(loader, model, optimizer, config, scheduler=None):
    # train one epoch
    losses = AverageMeter()
    true_final, pred_final = [], []

    model.train()
    train_iterator = tqdm(loader, leave=False)

    for X_batch, y_batch in train_iterator:
        X_batch = X_batch.to(config.device)
        y_batch = y_batch.to(config.device).type(torch.long)

        batch_size = X_batch.size(0)

        logit, prob = model(X_batch)

        loss = torch.nn.CrossEntropyLoss()(logit, y_batch.view(-1))
        losses.update(loss.item(), batch_size)

        true_final.append(y_batch.cpu())
        pred_final.append(prob.detach().cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_iterator.set_description(f"train ce:{losses.avg:.4f}")

    true_final = torch.cat(true_final, dim=0).view(-1).numpy()
    pred_final = torch.argmax(torch.cat(pred_final, dim=0), dim=1).numpy()
    train_acc = ((true_final == pred_final) * 1).mean()

    return losses.avg, train_acc


def train_one_epoch_wm(loader, wm_loader, model, optimizer, config):
    # train one epoch with poisoning data
    losses = AverageMeter()
    true_final, pred_final = [], []

    model.train()
    train_iterator = tqdm(loader, leave=False)
    wm_iterator = iter(wm_loader)

    for X_batch, y_batch in train_iterator:
        X_batch = X_batch.to(config.device)
        y_batch = y_batch.to(config.device)

        try:
            X_wm, y_wm = next(wm_iterator)
        except StopIteration:
            wm_iterator = iter(wm_loader)
            X_wm, y_wm = next(wm_iterator)

        X_wm = X_wm.to(config.device)
        y_wm = y_wm.to(config.device)

        X_batch = torch.cat([X_batch, X_wm], dim=0)
        y_batch = torch.cat([y_batch, y_wm], dim=0).type(torch.long)

        batch_size = X_batch.size(0)

        logit, prob = model(X_batch)

        loss = torch.nn.CrossEntropyLoss()(logit, y_batch.view(-1))
        losses.update(loss.item(), batch_size)

        true_final.append(y_batch[:-X_wm.size(0)].cpu())
        pred_final.append(prob[:-X_wm.size(0)].detach().cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_iterator.set_description(f"train ce:{losses.avg:.4f}")

    true_final = torch.cat(true_final, dim=0).view(-1).numpy()
    pred_final = torch.argmax(torch.cat(pred_final, dim=0), dim=1).numpy()
    train_acc = ((true_final == pred_final) * 1).mean()

    return losses.avg, train_acc


def train_one_epoch_at(loader, model, optimizer, adversary, config):
    # train one epoch
    losses = AverageMeter()
    true_final, pred_final = [], []

    model.train()
    train_iterator = tqdm(loader, leave=False)

    for X_batch, y_batch in train_iterator:
        X_batch = X_batch.to(config.device)
        y_batch = y_batch.to(config.device).type(torch.long)

        batch_size = X_batch.size(0)

        # X_batch = adversary.perturb(X_batch, y_batch)

        logit, prob = model(X_batch)

        loss = torch.nn.CrossEntropyLoss()(logit, y_batch.view(-1))
        losses.update(loss.item(), batch_size)

        true_final.append(y_batch.cpu())
        pred_final.append(prob.detach().cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_iterator.set_description(f"train ce:{losses.avg:.4f}")

    true_final = torch.cat(true_final, dim=0).view(-1).numpy()
    pred_final = torch.argmax(torch.cat(pred_final, dim=0), dim=1).numpy()
    train_acc = ((true_final == pred_final) * 1).mean()

    return losses.avg, train_acc


def valid_one_epoch(loader, model, config):
    # validate one epoch
    losses = AverageMeter()
    true_final, pred_final = [], []

    model.eval()
    valid_iterator = tqdm(loader, leave=False)

    for i, (X_batch, y_batch) in enumerate(valid_iterator):
        X_batch = X_batch.to(config.device)
        y_batch = y_batch.to(config.device).type(torch.long)

        batch_size = X_batch.size(0)

        with torch.no_grad():
            logit, prob = model(X_batch)
            loss = torch.nn.CrossEntropyLoss()(logit, y_batch.view(-1))

        losses.update(loss.item(), batch_size)

        true_final.append(y_batch.cpu())
        pred_final.append(prob.detach().cpu())

        losses.update(loss.item(), batch_size)

        valid_iterator.set_description(f"valid ce:{losses.avg:.4f}")

    true_final = torch.cat(true_final, dim=0).view(-1).numpy()
    pred_final = torch.argmax(torch.cat(pred_final, dim=0), dim=1).numpy()
    valid_acc = ((true_final == pred_final) * 1).mean()

    return losses.avg, valid_acc


def valid_one_epoch_at(loader, model, adversary, config):
    # validate one epoch
    losses = AverageMeter()
    true_final, pred_final = [], []

    model.eval()
    valid_iterator = tqdm(loader, leave=False)

    for i, (X_batch, y_batch) in enumerate(valid_iterator):
        X_batch = X_batch.to(config.device)
        y_batch = y_batch.to(config.device).type(torch.long)

        batch_size = X_batch.size(0)

        X_batch = adversary.perturb(X_batch, y_batch)

        with torch.no_grad():
            logit, prob = model(X_batch)
            loss = torch.nn.CrossEntropyLoss()(logit, y_batch.view(-1))

        losses.update(loss.item(), batch_size)

        true_final.append(y_batch.cpu())
        pred_final.append(prob.detach().cpu())

        losses.update(loss.item(), batch_size)

        valid_iterator.set_description(f"valid ce:{losses.avg:.4f}")

    true_final = torch.cat(true_final, dim=0).view(-1).numpy()
    pred_final = torch.argmax(torch.cat(pred_final, dim=0), dim=1).numpy()
    valid_acc = ((true_final == pred_final) * 1).mean()

    return losses.avg, valid_acc


def predict_samples(loader, model, config):
    # validate one epoch
    true_final, pred_final = [], []

    model.eval()
    valid_iterator = tqdm(loader, leave=False)

    for i, (X_batch, y_batch) in enumerate(valid_iterator):
        X_batch = X_batch.to(config.device)
        y_batch = y_batch.to(config.device).type(torch.long)

        with torch.no_grad():
            _, prob = model(X_batch)

        true_final.append(y_batch.cpu())
        pred_final.append(prob.detach().cpu())

    true_final = torch.cat(true_final, dim=0).view(-1).numpy()
    pred_final = torch.argmax(torch.cat(pred_final, dim=0), dim=1).numpy()

    return true_final, pred_final

