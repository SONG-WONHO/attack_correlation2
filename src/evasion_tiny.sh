#!/bin/bash

if [ $1 == 0 ]; then
  python scripts.py --gpu-id 0 --dataset tiny --arch resnet18 --attack-type fgsm --targeted --exp-ids 125,150,148,126,137,145,127,139,147 > sh.tiny.poison.targeted.fgsm.out
  python scripts.py --gpu-id 0 --dataset tiny --arch resnet18 --attack-type bim --targeted --exp-ids 125,150,148,126,137,145,127,139,147 > sh.tiny.poison.targeted.bim.out
  python scripts.py --gpu-id 0 --dataset tiny --arch resnet18 --attack-type fgsm --exp-ids 125,150,148,126,137,145,127,139,147 > sh.tiny.poison.untargeted.fgsm.out
elif [ $1 == 1 ]; then
  python scripts.py --gpu-id 1 --dataset tiny --arch resnet18 --attack-type fgsm --targeted --exp-ids 124,134,143,128,138 > sh.tiny.class.targeted.fgsm.out
  python scripts.py --gpu-id 1 --dataset tiny --arch resnet18 --attack-type fgsm --exp-ids 124,134,143,128,138 > sh.tiny.class.untargeted.fgsm.out
  python scripts.py --gpu-id 1 --dataset tiny --arch resnet18 --attack-type bim --targeted --exp-ids 124,134,143,128,138 > sh.tiny.class.targeted.bim.out
elif [ $1 == 2 ]; then
  python scripts.py --gpu-id 2 --dataset tiny --arch resnet18 --attack-type fgsm --targeted --exp-ids 146,129,136,144,130,132 > sh.tiny.mask.targeted.fgsm.out
  python scripts.py --gpu-id 2 --dataset tiny --arch resnet18 --attack-type fgsm --exp-ids 146,129,136,144,130,132 > sh.tiny.mask.untargeted.fgsm.out
  python scripts.py --gpu-id 2 --dataset tiny --arch resnet18 --attack-type bim --targeted --exp-ids 146,129,136,144,130,132 > sh.tiny.mask.targeted.bim.out
elif [ $1 == 3 ]; then
  python scripts.py --gpu-id 3 --dataset tiny --arch resnet18 --attack-type fgsm --targeted --exp-ids 140,131,133,141,151 > sh.tiny.size.targeted.fgsm.out
  python scripts.py --gpu-id 3 --dataset tiny --arch resnet18 --attack-type fgsm --exp-ids 140,131,133,141,151 > sh.tiny.size.untargeted.fgsm.out
  python scripts.py --gpu-id 3 --dataset tiny --arch resnet18 --attack-type bim --targeted --exp-ids 140,131,133,141,151 > sh.tiny.size.targeted.bim.out
elif [ $1 == 4 ]; then
  python scripts.py --gpu-id 4 --dataset tiny --arch resnet18 --attack-type bim --exp-ids 125,150,148,126,137,145,127,139,147 > sh.tiny.poison.untargeted.bim.out
  python scripts.py --gpu-id 4 --dataset tiny --arch resnet18 --attack-type pgd --targeted --exp-ids 140,131,133,141,151 > sh.tiny.size.targeted.pgd.out
  python scripts.py --gpu-id 4 --dataset tiny --arch resnet18 --attack-type pgd --exp-ids 140,131,133,141,151 > sh.tiny.size.untargeted.pgd.out
elif [ $1 == 5 ]; then
  python scripts.py --gpu-id 5 --dataset tiny --arch resnet18 --attack-type bim --exp-ids 140,131,133,141,151 > sh.tiny.size.untargeted.bim.out
  python scripts.py --gpu-id 5 --dataset tiny --arch resnet18 --attack-type pgd --targeted --exp-ids 146,129,136,144,130,132 > sh.tiny.mask.targeted.pgd.out
  python scripts.py --gpu-id 5 --dataset tiny --arch resnet18 --attack-type pgd --exp-ids 146,129,136,144,130,132 > sh.tiny.mask.untargeted.pgd.out
elif [ $1 == 6 ]; then
  python scripts.py --gpu-id 6 --dataset tiny --arch resnet18 --attack-type bim --exp-ids 146,129,136,144,130,132 > sh.tiny.mask.untargeted.bim.out
  python scripts.py --gpu-id 6 --dataset tiny --arch resnet18 --attack-type pgd --targeted --exp-ids 125,150,148,126,137,145,127,139,147 > sh.tiny.poison.targeted.pgd.out
  python scripts.py --gpu-id 6 --dataset tiny --arch resnet18 --attack-type pgd --exp-ids 125,150,148,126,137,145,127,139,147 > sh.tiny.poison.untargeted.pgd.out
elif [ $1 == 7 ]; then
  python scripts.py --gpu-id 7 --dataset tiny --arch resnet18 --attack-type bim --exp-ids 124,134,143,128,138 > sh.tiny.class.untargeted.bim.out
  python scripts.py --gpu-id 7 --dataset tiny --arch resnet18 --attack-type pgd --targeted --exp-ids 124,134,143,128,138 > sh.tiny.class.targeted.pgd.out
  python scripts.py --gpu-id 7 --dataset tiny --arch resnet18 --attack-type pgd --exp-ids 124,134,143,128,138 > sh.tiny.class.untargeted.pgd.out
fi