import os


class CFG:
    # path
    log_path = './log/attack/poison'
    model_path = './model/attack/poison'

    # data
    dataset = "mnist"

    # model
    arch = "lenet5"

    # learning
    batch_size = 64
    learning_rate = 1e-2
    momentum = 0.9
    num_epochs = 10

    # etc
    seed = 42
    worker = 1

    # backdoor
    pretrained_path = None
    backdoor_type = "blend"

    # factor default
    """
    poison ratio
        - MNIST: 1,2,3,4,5,6,7,8,9%
        - CIFAR10: 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3%
        - IMAGENET: 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,4,5,6,7,8,9%
    class ratio
        - ALL: 10,20,30,40,50%
    mask ratio
        - ALL: 5,10,20,40,80,100%
    size ratio
        - MNIST: 8,11,14,17,20px
        - CIFAR10: 6,7,8,9,10px
        - IMAGENET: 7,11,15,19,23px
    """


def main():
    """main function
    """

    ### header
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        choices=['mnist', 'cifar10', 'cifar100', 'aptos',
                                 'tiny'],
                        default=CFG.dataset,
                        help=f"Dataset({CFG.dataset})")
    parser.add_argument('--arch',
                        choices=['lenet5', 'resnet18', 'resnet34', 'resnet50'],
                        default=CFG.arch,
                        help=f"Architecture({CFG.arch})")

    # learning
    parser.add_argument('--batch-size',
                        default=CFG.batch_size,
                        type=int,
                        help=f"batch size({CFG.batch_size})")
    parser.add_argument('--learning-rate',
                        default=CFG.learning_rate,
                        type=float,
                        help=f"learning rate({CFG.learning_rate})")
    parser.add_argument('--num-epochs',
                        default=CFG.num_epochs,
                        type=int,
                        help=f"number of epochs({CFG.num_epochs})")

    # backdoor attack
    parser.add_argument('--pretrained-path', help="Target Model Path.")
    parser.add_argument('--backdoor-type', default=CFG.backdoor_type,
                        help="Type of backdoor attacks, blend or ssba")

    # evasion attack info
    parser.add_argument('--evasion-attack',
                        choices=['fgsm', 'bim', 'cw', 'pgd', 'spsa'],
                        required=True)
    parser.add_argument('--evasion-type',
                        choices=['targeted-np', 'targeted-p', 'untargeted'],
                        required=True)

    # etc
    parser.add_argument("--worker",
                        default=CFG.worker,
                        type=int,
                        help=f"number of worker({CFG.worker})")
    parser.add_argument("--seed",
                        default=CFG.seed,
                        type=int,
                        help=f"seed({CFG.seed})")

    args = parser.parse_args()

    info = {
        "mnist": {
            "targeted-np": {
                "fgsm": {
                    "poison_ratio": 2,
                    "class_ratio": 2,
                    "mask_ratio": 1,
                    "size_ratio": 2,},
                "bim": {
                    "poison_ratio": 7,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5,},
                "pgd": {
                    "poison_ratio": 7,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 4,},
                "cw": {
                    "poison_ratio": 8,
                    "class_ratio": 5,
                    "mask_ratio": 4,
                    "size_ratio": 5,},
                "spsa": {
                    "poison_ratio": 2,
                    "class_ratio": 2,
                    "mask_ratio": 1,
                    "size_ratio": 2,},
            },
            "targeted-p": {
                "fgsm": {
                    "poison_ratio": 7,
                    "class_ratio": 4,
                    "mask_ratio": 3,
                    "size_ratio": 5,},
                "bim": {
                    "poison_ratio": 7,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5,},
                "pgd": {
                    "poison_ratio": 6,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5,},
                "cw": {
                    "poison_ratio": 8,
                    "class_ratio": 5,
                    "mask_ratio": 4,
                    "size_ratio": 5,},
                "spsa": {
                    "poison_ratio": 7,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5,},
            },
            "untargeted": {
                "fgsm": {
                    "poison_ratio": 7,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5,},
                "bim": {
                    "poison_ratio": 6,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5,},
                "pgd": {
                    "poison_ratio": 9,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5,},
                "cw": {
                    "poison_ratio": 8,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5,},
                "spsa": {
                    "poison_ratio": 7,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5,},
            },
        },
        "cifar10": {
            "targeted-np": {
                "fgsm": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "bim": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "pgd": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "cw": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "spsa": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
            },
            "targeted-p": {
                "fgsm": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "bim": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "pgd": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "cw": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "spsa": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
            },
            "untargeted": {
                "fgsm": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "bim": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "pgd": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "cw": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "spsa": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
            },
        },
        "tiny": {
            "targeted-np": {
                "fgsm": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "bim": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "pgd": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "cw": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "spsa": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
            },
            "targeted-p": {
                "fgsm": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "bim": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "pgd": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "cw": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "spsa": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
            },
            "untargeted": {
                "fgsm": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "bim": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "pgd": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "cw": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
                "spsa": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0,},
            },
        },
    }

    settings = info[args.dataset][args.evasion_type][args.evasion_attack]

    print(settings)

    command = "python code/train_backdoor.py"
    command += f" --dataset {args.dataset}"
    command += f" --arch {args.arch}"
    command += f" --num-epochs {args.num_epochs}"
    command += f" --backdoor-type {args.backdoor_type}"
    # command += f" --poison-ratio {poison_ratio}"
    # command += f" --class-ratio {clss_ratio}"
    # command += f" --mask-ratio {mask_ratio}"
    # command += f" --size-ratio {size_ratio}"

    print(command)

    # os.system()
