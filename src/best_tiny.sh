#!/bin/bash

if [ $1 == 0 ]; then
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --evasion-attack fgsm --evasion-type targeted-np > sh.best.tiny.1151.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --evasion-attack fgsm --evasion-type targeted-p > sh.best.tiny.4222.out

elif [ $1 == 1 ]; then
CUDA_VISIBLE_DEVICES=1 python code/train_backdoor_best.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --evasion-attack fgsm --evasion-type untargeted > sh.best.tiny.9525.out

elif [ $1 == 2 ]; then
CUDA_VISIBLE_DEVICES=2 python code/train_backdoor_best.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --evasion-attack pgd --evasion-type targeted-p > sh.best.tiny.8525.out

elif [ $1 == 3 ]; then
CUDA_VISIBLE_DEVICES=3 python code/train_backdoor_best.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --evasion-attack spsa --evasion-type targeted-np > sh.best.tiny.2351.out
CUDA_VISIBLE_DEVICES=3 python code/train_backdoor_best.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --evasion-attack spsa --evasion-type targeted-p > sh.best.tiny.4125.out

elif [ $1 == 4 ]; then
CUDA_VISIBLE_DEVICES=4 python code/train_backdoor_best.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --evasion-attack cw --evasion-type targeted-p > sh.best.tiny.9535.out

elif [ $1 == 5 ]; then
CUDA_VISIBLE_DEVICES=5 python code/train_backdoor_best.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --evasion-attack bim --evasion-type targeted-np > sh.best.tiny.9424.out

elif [ $1 == 6 ]; then
CUDA_VISIBLE_DEVICES=6 python code/train_backdoor_best.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --evasion-attack pgd --evasion-type targeted-np > sh.best.tiny.9323.out

elif [ $1 == 7 ]; then
CUDA_VISIBLE_DEVICES=7 python code/train_backdoor_best.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --evasion-attack cw --evasion-type targeted-np > sh.best.tiny.9134.out

fi
