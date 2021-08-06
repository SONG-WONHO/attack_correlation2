#!/bin/bash

if [ $1 == 0 ]; then
  python scripts.py --gpu-id 0 --dataset mnist --arch lenet5 --attack-type fgsm --targeted --exp-ids 107,108,113,112,109,110,114,115,111 > sh.mnist.poison.targeted.fgsm.out
  python scripts.py --gpu-id 0 --dataset mnist --arch lenet5 --attack-type fgsm --exp-ids 107,108,113,112,109,110,114,115,111 > sh.mnist.poison.untargeted.fgsm.out
  python scripts.py --gpu-id 0 --dataset mnist --arch lenet5 --attack-type bim --targeted --exp-ids 107,108,113,112,109,110,114,115,111 > sh.mnist.poison.targeted.bim.out
  python scripts.py --gpu-id 0 --dataset mnist --arch lenet5 --attack-type bim --exp-ids 107,108,113,112,109,110,114,115,111 > sh.mnist.poison.untargeted.bim.out
  python scripts.py --gpu-id 0 --dataset mnist --arch lenet5 --attack-type pgd --targeted --exp-ids 107,108,113,112,109,110,114,115,111 > sh.mnist.poison.targeted.pgd.out
  python scripts.py --gpu-id 0 --dataset mnist --arch lenet5 --attack-type pgd --exp-ids 107,108,113,112,109,110,114,115,111 > sh.mnist.poison.untargeted.pgd.out
  python scripts.py --gpu-id 0 --dataset mnist --arch lenet5 --attack-type spsa --targeted --exp-ids 116,117,118,119,120 > sh.mnist.class.targeted.spsa.out
  python scripts.py --gpu-id 0 --dataset mnist --arch lenet5 --attack-type spsa --exp-ids 123,122,124,126,121,127 > sh.mnist.mask.untargeted.spsa.out
  python scripts.py --gpu-id 0 --dataset mnist --arch lenet5 --attack-type spsa --exp-ids 129,125,128,130,131 > sh.mnist.size.untargeted.spsa.out
if [ $1 == 1 ]; then
  python scripts.py --gpu-id 1 --dataset mnist --arch lenet5 --attack-type fgsm --targeted --exp-ids 116,117,118,119,120 > sh.mnist.class.targeted.fgsm.out
  python scripts.py --gpu-id 1 --dataset mnist --arch lenet5 --attack-type fgsm --exp-ids 116,117,118,119,120 > sh.mnist.class.untargeted.fgsm.out
  python scripts.py --gpu-id 1 --dataset mnist --arch lenet5 --attack-type bim --targeted --exp-ids 116,117,118,119,120 > sh.mnist.class.targeted.bim.out
  python scripts.py --gpu-id 1 --dataset mnist --arch lenet5 --attack-type bim --exp-ids 116,117,118,119,120 > sh.mnist.class.untargeted.bim.out
  python scripts.py --gpu-id 1 --dataset mnist --arch lenet5 --attack-type pgd --targeted --exp-ids 116,117,118,119,120 > sh.mnist.class.targeted.pgd.out
  python scripts.py --gpu-id 1 --dataset mnist --arch lenet5 --attack-type pgd --exp-ids 116,117,118,119,120 > sh.mnist.class.untargeted.pgd.out
  python scripts.py --gpu-id 1 --dataset mnist --arch lenet5 --attack-type cw --targeted --exp-ids 116,117,118,119,120 > sh.mnist.class.targeted.cw.out
  python scripts.py --gpu-id 1 --dataset mnist --arch lenet5 --attack-type cw --exp-ids 116,117,118,119,120 > sh.mnist.class.untargeted.cw.out
  python scripts.py --gpu-id 1 --dataset mnist --arch lenet5 --attack-type spsa --exp-ids 116,117,118,119,120 > sh.mnist.class.untargeted.spsa.out
if [ $1 == 2 ]; then
  python scripts.py --gpu-id 2 --dataset mnist --arch lenet5 --attack-type fgsm --targeted --exp-ids 123,122,124,126,121,127 > sh.mnist.mask.targeted.fgsm.out
  python scripts.py --gpu-id 2 --dataset mnist --arch lenet5 --attack-type fgsm --exp-ids 123,122,124,126,121,127 > sh.mnist.mask.untargeted.fgsm.out
  python scripts.py --gpu-id 2 --dataset mnist --arch lenet5 --attack-type bim --targeted --exp-ids 123,122,124,126,121,127 > sh.mnist.mask.targeted.bim.out
  python scripts.py --gpu-id 2 --dataset mnist --arch lenet5 --attack-type bim --exp-ids 123,122,124,126,121,127 > sh.mnist.mask.untargeted.bim.out
  python scripts.py --gpu-id 2 --dataset mnist --arch lenet5 --attack-type pgd --targeted --exp-ids 123,122,124,126,121,127 > sh.mnist.mask.targeted.pgd.out
  python scripts.py --gpu-id 2 --dataset mnist --arch lenet5 --attack-type pgd --exp-ids 123,122,124,126,121,127 > sh.mnist.mask.untargeted.pgd.out
  python scripts.py --gpu-id 2 --dataset mnist --arch lenet5 --attack-type cw --targeted --exp-ids 123,122,124,126,121,127 > sh.mnist.mask.targeted.cw.out
  python scripts.py --gpu-id 2 --dataset mnist --arch lenet5 --attack-type cw --exp-ids 123,122,124,126,121,127 > sh.mnist.mask.untargeted.cw.out
  python scripts.py --gpu-id 2 --dataset mnist --arch lenet5 --attack-type spsa --targeted --exp-ids 123,122,124,126,121,127 > sh.mnist.mask.targeted.spsa.out

if [ $1 == 3 ]; then
  python scripts.py --gpu-id 3 --dataset mnist --arch lenet5 --attack-type fgsm --targeted --exp-ids 129,125,128,130,131 > sh.mnist.size.targeted.fgsm.out
  python scripts.py --gpu-id 3 --dataset mnist --arch lenet5 --attack-type fgsm --exp-ids 129,125,128,130,131 > sh.mnist.size.untargeted.fgsm.out
  python scripts.py --gpu-id 3 --dataset mnist --arch lenet5 --attack-type bim --targeted --exp-ids 129,125,128,130,131 > sh.mnist.size.targeted.bim.out
  python scripts.py --gpu-id 3 --dataset mnist --arch lenet5 --attack-type bim --exp-ids 129,125,128,130,131 > sh.mnist.size.untargeted.bim.out
  python scripts.py --gpu-id 3 --dataset mnist --arch lenet5 --attack-type pgd --targeted --exp-ids 129,125,128,130,131 > sh.mnist.size.targeted.pgd.out
  python scripts.py --gpu-id 3 --dataset mnist --arch lenet5 --attack-type pgd --exp-ids 129,125,128,130,131 > sh.mnist.size.untargeted.pgd.out
  python scripts.py --gpu-id 3 --dataset mnist --arch lenet5 --attack-type cw --targeted --exp-ids 129,125,128,130,131 > sh.mnist.size.targeted.cw.out
  python scripts.py --gpu-id 3 --dataset mnist --arch lenet5 --attack-type cw --exp-ids 129,125,128,130,131 > sh.mnist.size.untargeted.cw.out
  python scripts.py --gpu-id 3 --dataset mnist --arch lenet5 --attack-type spsa --targeted --exp-ids 129,125,128,130,131 > sh.mnist.size.targeted.spsa.out
if [ $1 == 4 ]; then
  python scripts.py --gpu-id 4 --dataset mnist --arch lenet5 --attack-type cw --targeted --exp-ids 107,108,113,112,109,110,114,115,111 > sh.mnist.poison.targeted.cw.out
  python scripts.py --gpu-id 4 --dataset mnist --arch lenet5 --attack-type cw --exp-ids 107,108,113,112,109,110,114,115,111 > sh.mnist.poison.untargeted.cw.out
  python scripts.py --gpu-id 4 --dataset mnist --arch lenet5 --attack-type spsa --targeted --exp-ids 107,108,113,112,109,110,114,115,111 > sh.mnist.poison.targeted.spsa.out
  python scripts.py --gpu-id 4 --dataset mnist --arch lenet5 --attack-type spsa --exp-ids 107,108,113,112,109,110,114,115,111 > sh.mnist.poison.untargeted.spsa.out
fi