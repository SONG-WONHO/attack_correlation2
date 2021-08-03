import sys, os
import argparse

from subprocess import PIPE
from subprocess import Popen


class CFG:
    # path
    log_path = './log/attack/evasion'
    model_path = './model/attack/evasion'

    # data
    dataset = "mnist"

    # model
    arch = "lenet5"

    # attack
    attack_type = "fgsm"
    const = 0.03  # eps or c
    case = 0  # 0: target, 1: samples, 2: classes, 3: intensity, 4: size
    targeted = False
    poisoned = False

    # etc
    seed = 42
    worker = 1


# set virtual environment
COMMAND = '/home/win7785/Venv/ds/bin/python code/attack_evasion.py'

parser = argparse.ArgumentParser()

parser.add_argument('--gpu-id', default='0')

parser.add_argument('--dataset',
                    choices=['mnist', 'cifar10', 'cifar100', 'aptos',
                             'tiny'], default=CFG.dataset,
                    help=f"Dataset({CFG.dataset})")
parser.add_argument('--arch',
                    choices=['lenet5', 'resnet18', 'resnet34', 'resnet50'],
                    default=CFG.arch,
                    help=f"Architecture({CFG.arch})")

# evasion attack
parser.add_argument('--attack-type',
                    choices=['fgsm', 'bim', 'cw', 'pgd', 'spsa'],
                    default=CFG.attack_type,
                    help=f"Attack Type({CFG.attack_type})")
parser.add_argument("--targeted", action="store_true", default=CFG.targeted,
                    help=f"Targeted Evasion Attack?")
parser.add_argument("--exp-ids", default=None, required=True)
parser.add_argument("--seed",
                    default=CFG.seed,
                    type=int,
                    help=f"seed({CFG.seed})")

args = parser.parse_args()

COMMAND += ' --dataset %s' % args.dataset
COMMAND += ' --arch %s' % args.arch
COMMAND += ' --attack-type %s' % args.attack_type
COMMAND += f' --const {CFG.const}'
COMMAND += ' --case 0'
if args.targeted:
    COMMAND += ' --targeted'
COMMAND += f' --worker {CFG.worker}'
COMMAND += f' --seed {args.seed}'

os.environ['CUDA_VISIBLE_DEVICES'] = '%s' % args.gpu_id

print(COMMAND)
p = Popen(COMMAND.split(), cwd='./', bufsize=0,
          stdout=PIPE, stderr=PIPE)
stdout, stderr = p.communicate()

stdout = stdout.split()[-7:]
cand = []
for line in stdout:
    line = line.split(b',')
    c = float(line[0])
    sr = float(line[-1])
    if sr < 0.4:
        cand += [(c, sr)]

if len(cand) == 0:
    final_c = 2/255
else:
    cand = sorted(cand, key=lambda x: x[1], reverse=True)
    final_c = cand[0][0]
print(cand[0][0] * 255, cand[0][1])

COMMAND = COMMAND.replace('--case 0', '--case 1')
COMMAND = COMMAND.replace(f'--const {CFG.const}',
                          '--const {}'.format(final_c))
COMMAND += f" --exp-ids {args.exp_ids}"
print(COMMAND)

p = Popen(COMMAND.split(), cwd='./', bufsize=0,
          stdout=PIPE, stderr=PIPE)
stdout, stderr = p.communicate()

results = []
for line in stdout.split(b'\n'):
    # print(line.decode("utf-8"))
    if len(line.decode("utf-8").split(",")) == 5:
        results.append(line.decode("utf-8").split(",")[-1])

if args.targeted:
    COMMAND += ' --poisoned'
    print(COMMAND)

    p = Popen(COMMAND.split(), cwd='./', bufsize=0,
              stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()

    for line in stdout.split(b'\n'):
        # print(line.decode("utf-8"))
        if len(line.decode("utf-8").split(",")) == 5:
            results.append(line.decode("utf-8").split(",")[-1])

for r in results:
    print(r)
