#!/bin/bash

if [ $1 == 0 ]; then
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --evasion-attack fgsm --evasion-type targeted-np > sh.best.mnist.fgsm.0.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --evasion-attack fgsm --evasion-type targeted-p > sh.best.mnist.fgsm.1.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --evasion-attack fgsm --evasion-type untargeted > sh.best.mnist.fgsm.2.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --evasion-attack bim --evasion-type targeted-np > sh.best.mnist.fgsm.0.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --evasion-attack bim --evasion-type targeted-p > sh.best.mnist.fgsm.1.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --evasion-attack bim --evasion-type untargeted > sh.best.mnist.fgsm.2.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --evasion-attack pgd --evasion-type targeted-np > sh.best.mnist.fgsm.0.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --evasion-attack pgd --evasion-type targeted-p > sh.best.mnist.fgsm.1.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --evasion-attack pgd --evasion-type untargeted > sh.best.mnist.fgsm.2.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --evasion-attack cw --evasion-type targeted-np > sh.best.mnist.fgsm.0.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --evasion-attack cw --evasion-type targeted-p > sh.best.mnist.fgsm.1.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --evasion-attack cw --evasion-type untargeted > sh.best.mnist.fgsm.2.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --evasion-attack spsa --evasion-type targeted-np > sh.best.mnist.fgsm.0.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --evasion-attack spsa --evasion-type targeted-p > sh.best.mnist.fgsm.1.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor_best.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --evasion-attack spsa --evasion-type untargeted > sh.best.mnist.fgsm.2.out
fi