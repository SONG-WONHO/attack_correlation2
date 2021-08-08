#!/bin/bash

if [ $1 == 0 ]; then
CUDA_VISIBLE_DEVICES=0 python code/train_watermark.py --dataset mnist --arch lenet5 --learning-rate 1e-2 --num-epochs 20 --batch-size 64 --wm-batch-size 8 --wm-type content > sh.wm.mnist.content.out
CUDA_VISIBLE_DEVICES=0 python code/train_watermark.py --dataset cifar10 --arch resnet18 --learning-rate 1e-2 --num-epochs 100 --batch-size 64 --wm-batch-size 8 --wm-type content > sh.wm.cifar10.content.out
CUDA_VISIBLE_DEVICES=0 python code/train_watermark.py --dataset tiny --arch resnet18 --learning-rate 1e-2 --num-epochs 100 --batch-size 64 --wm-batch-size 8 --wm-type content > sh.wm.tiny.content.out
elif [ $1 == 1 ]; then
CUDA_VISIBLE_DEVICES=1 python code/train_watermark.py --dataset mnist --arch lenet5 --learning-rate 1e-2 --num-epochs 20 --batch-size 64 --wm-batch-size 8 --wm-type noise > sh.wm.mnist.noise.out
CUDA_VISIBLE_DEVICES=1 python code/train_watermark.py --dataset cifar10 --arch resnet18 --learning-rate 1e-2 --num-epochs 100 --batch-size 64 --wm-batch-size 8 --wm-type noise > sh.wm.cifar10.noise.out
CUDA_VISIBLE_DEVICES=1 python code/train_watermark.py --dataset tiny --arch resnet18 --learning-rate 1e-2 --num-epochs 100 --batch-size 64 --wm-batch-size 8 --wm-type noise > sh.wm.tiny.noise.out
elif [ $1 == 2 ]; then
CUDA_VISIBLE_DEVICES=2 python code/train_watermark.py --dataset mnist --arch lenet5 --learning-rate 1e-2 --num-epochs 20 --batch-size 64 --wm-batch-size 8 --wm-type unrelate > sh.wm.mnist.unrelate.out
CUDA_VISIBLE_DEVICES=2 python code/train_watermark.py --dataset cifar10 --arch resnet18 --learning-rate 1e-2 --num-epochs 100 --batch-size 64 --wm-batch-size 8 --wm-type unrelate > sh.wm.cifar10.unrelate.out
CUDA_VISIBLE_DEVICES=2 python code/train_watermark.py --dataset tiny --arch resnet18 --learning-rate 1e-2 --num-epochs 100 --batch-size 64 --wm-batch-size 8 --wm-type unrelate > sh.wm.tiny.unrelate.out
elif [ $1 == 3 ]; then
CUDA_VISIBLE_DEVICES=3 python code/train_watermark.py --dataset mnist --arch lenet5 --learning-rate 1e-2 --num-epochs 20 --batch-size 64 --wm-batch-size 8 --wm-type abstract > sh.wm.mnist.abstract.out
CUDA_VISIBLE_DEVICES=3 python code/train_watermark.py --dataset cifar10 --arch resnet18 --learning-rate 1e-2 --num-epochs 100 --batch-size 64 --wm-batch-size 8 --wm-type abstract > sh.wm.cifar10.abstract.out
CUDA_VISIBLE_DEVICES=3 python code/train_watermark.py --dataset tiny --arch resnet18 --learning-rate 1e-2 --num-epochs 100 --batch-size 64 --wm-batch-size 8 --wm-type abstract > sh.wm.tiny.abstract.out
fi