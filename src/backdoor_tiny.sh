#!/bin/bash

if [ $1 == 0 ]; then
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --class-ratio 0.1 > sh.tiny.class_ratio_0.1.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend  --class-ratio 0.2 > sh.tiny.class_ratio_0.2.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend  --class-ratio 0.3 > sh.tiny.class_ratio_0.3.out
CUDA_VISIBLE_DEVICES=0 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --size-ratio 24 > sh.tiny.size_ratio_24.out

elif [ $1 == 1 ]; then
CUDA_VISIBLE_DEVICES=1 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --poison-ratio 0.01 > sh.tiny.poison_ratio_0.01.out
CUDA_VISIBLE_DEVICES=1 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --poison-ratio 0.02 > sh.tiny.poison_ratio_0.02.out
CUDA_VISIBLE_DEVICES=1 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --poison-ratio 0.04 > sh.tiny.poison_ratio_0.04.out
CUDA_VISIBLE_DEVICES=1 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --size-ratio 20 > sh.tiny.size_ratio_20.out

elif [ $1 == 2 ]; then
CUDA_VISIBLE_DEVICES=2 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --poison-ratio 0.06 > sh.tiny.poison_ratio_0.06.out
CUDA_VISIBLE_DEVICES=2 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --poison-ratio 0.08 > sh.tiny.poison_ratio_0.08.out
CUDA_VISIBLE_DEVICES=2 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --poison-ratio 0.10 > sh.tiny.poison_ratio_0.1.out

elif [ $1 == 3 ]; then
CUDA_VISIBLE_DEVICES=3 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --poison-ratio 0.15 > sh.tiny.poison_ratio_0.15.out
CUDA_VISIBLE_DEVICES=3 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --poison-ratio 0.20 > sh.tiny.poison_ratio_0.20.out
CUDA_VISIBLE_DEVICES=3 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --poison-ratio 0.30 > sh.tiny.poison_ratio_0.30.out

elif [ $1 == 4 ]; then
CUDA_VISIBLE_DEVICES=4 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend  --class-ratio 0.4 > sh.tiny.class_ratio_0.4.out
CUDA_VISIBLE_DEVICES=4 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend  --class-ratio 0.5 > sh.tiny.class_ratio_0.5.out
CUDA_VISIBLE_DEVICES=4 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --mask-ratio 0.05 > sh.tiny.mask_ratio_0.05.out

elif [ $1 == 5 ]; then
CUDA_VISIBLE_DEVICES=5 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --mask-ratio 0.10 > sh.tiny.mask_ratio_0.10.out
CUDA_VISIBLE_DEVICES=5 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --mask-ratio 0.20 > sh.tiny.mask_ratio_0.20.out
CUDA_VISIBLE_DEVICES=5 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --mask-ratio 0.40 > sh.tiny.mask_ratio_0.40.out

elif [ $1 == 6 ]; then
CUDA_VISIBLE_DEVICES=6 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --mask-ratio 0.80 > sh.tiny.mask_ratio_0.80.out
CUDA_VISIBLE_DEVICES=6 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --mask-ratio 1.00 > sh.tiny.mask_ratio_1.00.out
CUDA_VISIBLE_DEVICES=6 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --size-ratio 4 > sh.tiny.size_ratio_4.out

elif [ $1 == 7 ]; then
CUDA_VISIBLE_DEVICES=7 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --size-ratio 8 > sh.tiny.size_ratio_8.out
CUDA_VISIBLE_DEVICES=7 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --size-ratio 12 > sh.tiny.size_ratio_12.out
CUDA_VISIBLE_DEVICES=7 python code/train_backdoor.py --dataset tiny --arch resnet18 --num-epochs 100 --backdoor-type blend --size-ratio 16 > sh.tiny.size_ratio_16.out

fi
