#!/bin/bash

if [ $1 == 4 ]; then
CUDA_VISIBLE_DEVICES=4 python code/train_backdoor_best.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --evasion-attack fgsm --evasion-type targeted-np > sh.best.cifar10.3242.out
CUDA_VISIBLE_DEVICES=4 python code/train_backdoor_best.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --evasion-attack bim --evasion-type targeted-np > sh.best.cifar10.6526.out
CUDA_VISIBLE_DEVICES=4 python code/train_backdoor_best.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --evasion-attack fgsm --evasion-type targeted-p > sh.best.cifar10.8425.out
CUDA_VISIBLE_DEVICES=4 python code/train_backdoor_best.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --evasion-attack pgd --evasion-type targeted-p > sh.best.cifar10.9525.out
CUDA_VISIBLE_DEVICES=4 python code/train_backdoor_best.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --evasion-attack cw --evasion-type targeted-p > sh.best.cifar10.8525.out
#elif [ $1 == 1 ]; then
#
#elif [ $1 == 2 ]; then
#
#
#elif [ $1 == 4 ]; then

fi
