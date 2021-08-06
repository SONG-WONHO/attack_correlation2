import os
import argparse


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
                    "size_ratio": 2, },
                "bim": {
                    "poison_ratio": 7,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5, },
                "pgd": {
                    "poison_ratio": 7,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 4, },
                "cw": {
                    "poison_ratio": 8,
                    "class_ratio": 5,
                    "mask_ratio": 4,
                    "size_ratio": 5, },
                "spsa": {
                    "poison_ratio": 2,
                    "class_ratio": 2,
                    "mask_ratio": 1,
                    "size_ratio": 2, },
            },
            "targeted-p": {
                "fgsm": {
                    "poison_ratio": 7,
                    "class_ratio": 4,
                    "mask_ratio": 3,
                    "size_ratio": 5, },
                "bim": {
                    "poison_ratio": 7,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5, },
                "pgd": {
                    "poison_ratio": 6,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5, },
                "cw": {
                    "poison_ratio": 8,
                    "class_ratio": 5,
                    "mask_ratio": 4,
                    "size_ratio": 5, },
                "spsa": {
                    "poison_ratio": 7,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5, },
            },
            "untargeted": {
                "fgsm": {
                    "poison_ratio": 7,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5, },
                "bim": {
                    "poison_ratio": 6,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5, },
                "pgd": {
                    "poison_ratio": 9,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5, },
                "cw": {
                    "poison_ratio": 8,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5, },
                "spsa": {
                    "poison_ratio": 7,
                    "class_ratio": 5,
                    "mask_ratio": 3,
                    "size_ratio": 5, },
            },
        },
        "cifar10": {
            "targeted-np": {
                "fgsm": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "bim": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "pgd": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "cw": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "spsa": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
            },
            "targeted-p": {
                "fgsm": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "bim": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "pgd": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "cw": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "spsa": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
            },
            "untargeted": {
                "fgsm": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "bim": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "pgd": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "cw": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "spsa": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
            },
        },
        "tiny": {
            "targeted-np": {
                "fgsm": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "bim": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "pgd": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "cw": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "spsa": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
            },
            "targeted-p": {
                "fgsm": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "bim": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "pgd": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "cw": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "spsa": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
            },
            "untargeted": {
                "fgsm": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "bim": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "pgd": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "cw": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
                "spsa": {
                    "poison_ratio": 0,
                    "class_ratio": 0,
                    "mask_ratio": 0,
                    "size_ratio": 0, },
            },
        },
    }

    mp = {
        "poison_ratio": {
            "mnist": {
                1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05, 6: 0.06, 7: 0.07,
                8: 0.08, 9: 0.09,
            },
            "cifar10": {
                1: 0.01, 2: 0.0125, 3: 0.015, 4: 0.0175, 5: 0.02, 6: 0.0225,
                7: 0.025, 8: 0.0275, 9: 0.03
            },
            "tiny": {
                1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05, 6: 0.06, 7: 0.07,
                8: 0.08, 9: 0.09,
            },
        },
        "class_ratio": {
            "mnist": {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5},
            "cifar10": {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5},
            "imagenet": {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5},
        },
        "mask_ratio": {
            "mnist": {1: 0.05, 2: 0.1, 3: 0.2, 4: 0.4, 5: 0.8, 6: 1.0},
            "cifar10": {1: 0.05, 2: 0.1, 3: 0.2, 4: 0.4, 5: 0.8, 6: 1.0},
            "imagenet": {1: 0.05, 2: 0.1, 3: 0.2, 4: 0.4, 5: 0.8, 6: 1.0}
        },
        "size_ratio": {
            "mnist": {1: 8, 2: 11, 3: 14, 4: 17, 5: 20},
            "cifar10": {1: 6, 2: 7, 3: 8, 4: 9, 5: 10},
            "imagenet": {1: 7, 2: 11, 3: 15, 4: 19, 5: 23},
        }
    }

    levels = info[args.dataset][args.evasion_type][args.evasion_attack]

    poison_ratio = mp['poison_ratio'][args.dataset][levels['poison_ratio']]
    class_ratio = mp['class_ratio'][args.dataset][levels['class_ratio']]
    mask_ratio = mp['mask_ratio'][args.dataset][levels['mask_ratio']]
    size_ratio = mp['size_ratio'][args.dataset][levels['size_ratio']]

    command = "python code/train_backdoor.py"
    command += f" --dataset {args.dataset}"
    command += f" --arch {args.arch}"
    command += f" --num-epochs {args.num_epochs}"
    command += f" --backdoor-type {args.backdoor_type}"
    command += f" --poison-ratio {poison_ratio}"
    command += f" --class-ratio {clss_ratio}"
    command += f" --mask-ratio {mask_ratio}"
    command += f" --size-ratio {size_ratio}"

    print(command)

    # os.system()


if __name__ == "__main__":
    main()
