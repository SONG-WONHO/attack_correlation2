#!/bin/bash

if [ $1 == 0 ]; then
  python scripts.py --gpu-id 0 --dataset cifar10 --arch resnet18 --attack-type fgsm --targeted --exp-ids 122,126,128,131,123,127,129,132,133 > sh.cifar10.poison.targeted.fgsm.out
  python scripts.py --gpu-id 0 --dataset cifar10 --arch resnet18 --attack-type fgsm --exp-ids 122,126,128,131,123,127,129,132,133 > sh.cifar10.poison.untargeted.fgsm.out
  python scripts.py --gpu-id 0 --dataset cifar10 --arch resnet18 --attack-type bim --targeted --exp-ids 122,126,128,131,123,127,129,132,133 > sh.cifar10.poison.targeted.bim.out
  python scripts.py --gpu-id 0 --dataset cifar10 --arch resnet18 --attack-type pgd --targeted --exp-ids 122,126,128,131,123,127,129,132,133 > sh.cifar10.poison.targeted.pgd.out
  python scripts.py --gpu-id 0 --dataset cifar10 --arch resnet18 --attack-type pgd --exp-ids 122,126,128,131,123,127,129,132,133 > sh.cifar10.poison.untargeted.pgd.out
elif [ $1 == 1 ]; then
  python scripts.py --gpu-id 1 --dataset cifar10 --arch resnet18 --attack-type fgsm --targeted --exp-ids 135,138,134,136,124 > sh.cifar10.class.targeted.fgsm.out
  python scripts.py --gpu-id 1 --dataset cifar10 --arch resnet18 --attack-type fgsm --exp-ids 135,138,134,136,124 > sh.cifar10.class.untargeted.fgsm.out
  python scripts.py --gpu-id 1 --dataset cifar10 --arch resnet18 --attack-type bim --targeted --exp-ids 135,138,134,136,124 > sh.cifar10.class.targeted.bim.out
  python scripts.py --gpu-id 1 --dataset cifar10 --arch resnet18 --attack-type pgd --targeted --exp-ids 135,138,134,136,124 > sh.cifar10.class.targeted.pgd.out
  python scripts.py --gpu-id 1 --dataset cifar10 --arch resnet18 --attack-type pgd --exp-ids 135,138,134,136,124 > sh.cifar10.class.untargeted.pgd.out
elif [ $1 == 2 ]; then
  python scripts.py --gpu-id 2 --dataset cifar10 --arch resnet18 --attack-type fgsm --targeted --exp-ids 130,137,141,121,139,142 > sh.cifar10.mask.targeted.fgsm.out
  python scripts.py --gpu-id 2 --dataset cifar10 --arch resnet18 --attack-type fgsm --exp-ids 130,137,141,121,139,142 > sh.cifar10.mask.untargeted.fgsm.out
  python scripts.py --gpu-id 2 --dataset cifar10 --arch resnet18 --attack-type bim --targeted --exp-ids 130,137,141,121,139,142 > sh.cifar10.mask.targeted.bim.out
  python scripts.py --gpu-id 2 --dataset cifar10 --arch resnet18 --attack-type pgd --targeted --exp-ids 130,137,141,121,139,142 > sh.cifar10.mask.targeted.pgd.out
  python scripts.py --gpu-id 2 --dataset cifar10 --arch resnet18 --attack-type pgd --exp-ids 130,137,141,121,139,142 > sh.cifar10.mask.untargeted.pgd.out
elif [ $1 == 3 ]; then
  python scripts.py --gpu-id 3 --dataset cifar10 --arch resnet18 --attack-type fgsm --targeted --exp-ids 125,140,145,144,143 > sh.cifar10.size.targeted.fgsm.out
  python scripts.py --gpu-id 3 --dataset cifar10 --arch resnet18 --attack-type fgsm --exp-ids 125,140,145,144,143 > sh.cifar10.size.untargeted.fgsm.out
  python scripts.py --gpu-id 3 --dataset cifar10 --arch resnet18 --attack-type bim --targeted --exp-ids 125,140,145,144,143 > sh.cifar10.size.targeted.bim.out
  python scripts.py --gpu-id 3 --dataset cifar10 --arch resnet18 --attack-type pgd --targeted --exp-ids 125,140,145,144,143 > sh.cifar10.size.targeted.pgd.out
  python scripts.py --gpu-id 3 --dataset cifar10 --arch resnet18 --attack-type pgd --exp-ids 125,140,145,144,143 > sh.cifar10.size.untargeted.pgd.out
elif [ $1 == 4 ]; then
  python scripts.py --gpu-id 4 --dataset cifar10 --arch resnet18 --attack-type bim --exp-ids 122,126,128,131,123,127,129,132,133 > sh.cifar10.poison.untargeted.bim.out
  python scripts.py --gpu-id 4 --dataset cifar10 --arch resnet18 --attack-type bim --exp-ids 135,138,134,136,124 > sh.cifar10.class.untargeted.bim.out
  python scripts.py --gpu-id 4 --dataset cifar10 --arch resnet18 --attack-type bim --exp-ids 130,137,141,121,139,142 > sh.cifar10.mask.untargeted.bim.out
  python scripts.py --gpu-id 4 --dataset cifar10 --arch resnet18 --attack-type bim --exp-ids 125,140,145,144,143 > sh.cifar10.size.untargeted.bim.out
fi
