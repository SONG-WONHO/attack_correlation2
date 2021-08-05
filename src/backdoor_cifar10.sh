#!/bin/bash

if [ $1 == 0 ]; then
  CUDA_VISIBLE_DEVICES=0 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --mask-ratio 0.40 > sh.cifar10.mask_ratio_0.40.out
  CUDA_VISIBLE_DEVICES=0 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --mask-ratio 0.80 > sh.cifar10.mask_ratio_0.80.out
  CUDA_VISIBLE_DEVICES=0 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --mask-ratio 1.00 > sh.cifar10.mask_ratio_1.00.out
  CUDA_VISIBLE_DEVICES=0 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --size-ratio 9 > sh.cifar10.size_ratio_9.out
elif [ $1 == 1 ]; then
  CUDA_VISIBLE_DEVICES=1 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --poison-ratio 0.01 > sh.cifar10.poison_ratio_0.01.out
  CUDA_VISIBLE_DEVICES=1 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --poison-ratio 0.0125 > sh.cifar10.poison_ratio_0.0125.out
  CUDA_VISIBLE_DEVICES=1 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --poison-ratio 0.015 > sh.cifar10.poison_ratio_0.015.out
  CUDA_VISIBLE_DEVICES=1 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --poison-ratio 0.0175 > sh.cifar10.poison_ratio_0.0175.out
  CUDA_VISIBLE_DEVICES=1 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --poison-ratio 0.03 > sh.cifar10.poison_ratio_0.03.out
  CUDA_VISIBLE_DEVICES=1 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --class-ratio 0.1 > sh.cifar10.class_ratio_0.1.out
  CUDA_VISIBLE_DEVICES=1 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend  --class-ratio 0.2 > sh.cifar10.class_ratio_0.2.out
elif [ $1 == 2 ]; then
  CUDA_VISIBLE_DEVICES=2 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --poison-ratio 0.02 > sh.cifar10.poison_ratio_0.02.out
  CUDA_VISIBLE_DEVICES=2 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --poison-ratio 0.0225 > sh.cifar10.poison_ratio_0.0225.out
  CUDA_VISIBLE_DEVICES=2 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --poison-ratio 0.025 > sh.cifar10.poison_ratio_0.025.out
  CUDA_VISIBLE_DEVICES=2 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --poison-ratio 0.0275 > sh.cifar10.poison_ratio_0.0275.out
  CUDA_VISIBLE_DEVICES=2 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend  --class-ratio 0.3 > sh.cifar10.class_ratio_0.3.out
  CUDA_VISIBLE_DEVICES=2 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend  --class-ratio 0.4 > sh.cifar10.class_ratio_0.4.out
elif [ $1 == 3 ]; then
  CUDA_VISIBLE_DEVICES=3 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend  --class-ratio 0.5 > sh.cifar10.class_ratio_0.5.out
  CUDA_VISIBLE_DEVICES=3 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --mask-ratio 0.05 > sh.cifar10.mask_ratio_0.05.out
  CUDA_VISIBLE_DEVICES=3 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --mask-ratio 0.10 > sh.cifar10.mask_ratio_0.10.out
  CUDA_VISIBLE_DEVICES=3 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --mask-ratio 0.20 > sh.cifar10.mask_ratio_0.20.out
  CUDA_VISIBLE_DEVICES=3 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --size-ratio 10 > sh.cifar10.size_ratio_10.out
elif [ $1 == 4 ]; then
  CUDA_VISIBLE_DEVICES=4 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --size-ratio 6 > sh.cifar10.size_ratio_6.out
  CUDA_VISIBLE_DEVICES=4 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --size-ratio 7 > sh.cifar10.size_ratio_7.out
  CUDA_VISIBLE_DEVICES=4 python code/train_backdoor.py --dataset cifar10 --arch resnet18 --num-epochs 60 --backdoor-type blend --size-ratio 8 > sh.cifar10.size_ratio_8.out

fi