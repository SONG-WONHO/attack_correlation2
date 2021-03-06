#!/bin/bash

if [ $1 == 0 ]; then
  CUDA_VISIBLE_DEVICES=0 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --poison-ratio 0.01 > sh.mnist.poison_ratio_0.01.out
  CUDA_VISIBLE_DEVICES=0 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --poison-ratio 0.04 > sh.mnist.poison_ratio_0.04.out
  CUDA_VISIBLE_DEVICES=0 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend  --class-ratio 0.2 > sh.mnist.class_ratio_0.2.out
  CUDA_VISIBLE_DEVICES=0 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --mask-ratio 0.05 > sh.mnist.mask_ratio_0.05.out

elif [ $1 == 1 ]; then
  CUDA_VISIBLE_DEVICES=1 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --poison-ratio 0.02 > sh.mnist.poison_ratio_0.02.out
  CUDA_VISIBLE_DEVICES=1 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --poison-ratio 0.03 > sh.mnist.poison_ratio_0.03.out
  CUDA_VISIBLE_DEVICES=1 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend  --class-ratio 0.3 > sh.mnist.class_ratio_0.3.out
  CUDA_VISIBLE_DEVICES=1 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --mask-ratio 0.10 > sh.mnist.mask_ratio_0.10.out

elif [ $1 == 2 ]; then
  CUDA_VISIBLE_DEVICES=2 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --poison-ratio 0.05 > sh.mnist.poison_ratio_0.05.out
  CUDA_VISIBLE_DEVICES=2 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --poison-ratio 0.08 > sh.mnist.poison_ratio_0.08.out
  CUDA_VISIBLE_DEVICES=2 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend  --class-ratio 0.4 > sh.mnist.class_ratio_0.4.out
  CUDA_VISIBLE_DEVICES=2 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --mask-ratio 0.20 > sh.mnist.mask_ratio_0.20.out
  CUDA_VISIBLE_DEVICES=2 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --mask-ratio 1.00 > sh.mnist.mask_ratio_1.00.out

elif [ $1 == 3 ]; then
  CUDA_VISIBLE_DEVICES=3 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --poison-ratio 0.06 > sh.mnist.poison_ratio_0.06.out
  CUDA_VISIBLE_DEVICES=3 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --poison-ratio 0.07 > sh.mnist.poison_ratio_0.07.out
  CUDA_VISIBLE_DEVICES=3 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend  --class-ratio 0.5 > sh.mnist.class_ratio_0.5.out
  CUDA_VISIBLE_DEVICES=3 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --mask-ratio 0.40 > sh.mnist.mask_ratio_0.40.out
  CUDA_VISIBLE_DEVICES=3 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --size-ratio 8 > sh.mnist.size_ratio_8.out
  CUDA_VISIBLE_DEVICES=3 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --size-ratio 17 > sh.mnist.size_ratio_17.out

elif [ $1 == 4 ]; then
  CUDA_VISIBLE_DEVICES=4 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --poison-ratio 0.09 > sh.mnist.poison_ratio_0.09.out
  CUDA_VISIBLE_DEVICES=4 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --class-ratio 0.1 > sh.mnist.class_ratio_0.1.out
  CUDA_VISIBLE_DEVICES=4 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --mask-ratio 0.80 > sh.mnist.mask_ratio_0.80.out
  CUDA_VISIBLE_DEVICES=4 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --size-ratio 11 > sh.mnist.size_ratio_11.out
  CUDA_VISIBLE_DEVICES=4 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --size-ratio 14 > sh.mnist.size_ratio_14.out
  CUDA_VISIBLE_DEVICES=4 python code/train_backdoor.py --dataset mnist --arch lenet5 --num-epochs 20 --backdoor-type blend --size-ratio 20 > sh.mnist.size_ratio_20.out

fi