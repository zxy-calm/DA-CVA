import datetime
import csv
import json
import os
import random
from termcolor import colored

import torch
import numpy as np
import argparse

from models import MyModels
from get_dataloader import _get_loader
from train import trainer
from test import test

import warnings
warnings.filterwarnings("ignore")


np.set_printoptions(suppress=True)


parser = argparse.ArgumentParser(description="medical image classification")

parser.add_argument("--dataset", type=str, default="chestct", help="dataset: [esophageal, gastric, ham10000, chestct]")
parser.add_argument("--shot", type=int, default=30, help="[5, 15, 30] for few-shot")

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--train_batch_size", type=int, default=20)
parser.add_argument("--test_batch_size", type=int, default=40)

parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")
"""
weight_decay: esophageal[5: 0.001, 15: 0.0001: 30: 0.00001], gastric[5: 0.01, 15: 0.01: 30: 0.0001], 
ham10000[5: 0.0001, 15: 0.00001: 30: 0.00001], chestct[5: 0.01, 15: 0.01: 30: 0.1]
"""
parser.add_argument("--beta", type=float, default=0.3)

parser.set_defaults(use_class_prior_kl=True)
parser.add_argument(
    "--use_class_prior_kl",
    dest="use_class_prior_kl",
    action="store_true",
    help="Use class-conditional prior KL: KL(q(z|x)||p(z|y))"
)
parser.add_argument(
    "--no_use_class_prior_kl",
    dest="use_class_prior_kl",
    action="store_false",
    help="Use standard normal prior KL: KL(q(z|x)||N(0, I))"
)


parser.add_argument("--mode", type=str, default="test", help="[train, test]")
parser.add_argument("--patience", type=int, default=20, help="early stop")
parser.add_argument("--seed", type=int, default=107, help="Seed setting [104, 111, 101, 107]")
parser.add_argument("--cuda", type=int, default=0, help="device, -1 for cpu, 0,1,... for cuda")
parser.add_argument("--checkpoint", type=bool, default=True)
parser.add_argument("--result_path", type=str, default="results.json")
parser.add_argument("--ablate_experts", action="store_true", help="Ablate expert branches while keeping the ConvNeXt main branch trainable")
parser.add_argument("--ablate_backbone", action="store_true", help="Ablate the ConvNeXt backbone branch and use expert branches only")
parser.add_argument("--ablate_routing", nargs="+", choices=("mu", "std", "emb", "all"), default=[],
                    help="Replace selected expert routing with uniform averaging: mu, std, emb, or all")

args = parser.parse_args()

if args.ablate_experts and args.ablate_backbone:
    raise ValueError("--ablate_experts and --ablate_backbone cannot be used together")
if args.ablate_routing and (args.ablate_experts or args.ablate_backbone):
    parser.error("--ablate_routing requires both backbone and expert branches; do not combine with --ablate_experts or --ablate_backbone")


def set_seed(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_param_(args):

    if args.dataset == "esophageal":
        if args.shot == 5:
            setattr(args, 'weight_decay', 0.001)
        elif args.shot == 15:
            setattr(args, 'weight_decay', 0.0001)
        elif args.shot == 30:
            setattr(args, 'weight_decay', 0.00001)
    elif args.dataset == "gastric":
        if args.shot == 5:
            setattr(args, 'weight_decay', 0.01)
        elif args.shot == 15:
            setattr(args, 'weight_decay', 0.01)
        elif args.shot == 30:
            setattr(args, 'weight_decay', 0.00001)
    elif args.dataset == "ham10000":
        if args.shot == 5:
            setattr(args, 'weight_decay', 0.0001)
        elif args.shot == 15:
            setattr(args, 'weight_decay', 0.0001)
        elif args.shot == 30:
            setattr(args, 'weight_decay', 0.00001)
    elif args.dataset == "chestct":
        if args.shot == 5:
            setattr(args, 'weight_decay', 0.1)
        elif args.shot == 15:
            setattr(args, 'weight_decay', 0.01)
        elif args.shot == 30:
            setattr(args, 'weight_decay', 0.1)
    else:
        raise NotImplementedError


def get_device(args):
    device = f"cuda:{args.cuda}" if args.cuda != -1 else "cpu"
    print("{}, operating on {:s}".format(
        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
        device
    ), flush=True)
    device = torch.device(device)
    torch.cuda.set_device(device)
    return device


def get_result_dir(args):
    result_root = os.path.splitext(args.result_path)[0] if args.result_path else "results"
    os.makedirs(result_root, exist_ok=True)
    return result_root


def get_dataset_result_path(dataset, args):
    result_dir = get_result_dir(args)
    return os.path.join(result_dir, f"{dataset}_acc_f1.csv")


def write_dataset_results(dataset, results, args):
    result_path = get_dataset_result_path(dataset, args)
    rows = sorted(results, key=lambda item: (item["shot"], item["seed"]))
    fieldnames = ["dataset", "shot", "seed", "acc", "f1", "best_epoch", "stop_epoch"]

    with open(result_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(seed=None, dataset=None, shot=None):
    if seed != None:
        setattr(args, 'seed', seed)
    set_seed(args.seed)

    if dataset != None and shot != None:
        setattr(args, 'dataset', dataset)
        setattr(args, 'shot', shot)

    print("{}, Mode: {:s}, Dataset: {:s}, Shot: {}, Random initial seed: {}".format(
        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'), args.mode, args.dataset, args.shot, args.seed
    ), flush=True)
    print(
        "{}, KL mode: {}".format(
            datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
            "class-conditional prior" if args.use_class_prior_kl else "standard normal prior"
        ),
        flush=True
    )
    
    set_param_(args)
    device = get_device(args)

    train_dataloader, val_dataloader, test_dataloader, id2label, num_classes = _get_loader(args, device)
    model = MyModels(num_classes=num_classes)
    model.ablate_routing = ({"mu", "std", "emb"} if "all" in args.ablate_routing
                            else set(args.ablate_routing))
    if model.ablate_routing:
        print("Routing ablation (uniform averaging): " + ", ".join(sorted(model.ablate_routing)), flush=True)
    if args.ablate_experts:
        model.ablate_experts = True
    elif args.ablate_backbone:
        model.ablate_backbone = True
    if args.cuda != -1:
        model = model.to(device)

    if args.mode == "train":
        trainer(model, train_dataloader, val_dataloader, args)
        print("{}, Training finished!".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')), flush=True)
    if args.dataset == "esophageal":
        classes_name = ["Cancer", "OES", "HGME", "EV", "ESB"]
    elif args.dataset == "gastric":
        classes_name = ["Cancer", "Ulcer", "Polyps", "ST"]
    elif args.dataset == "ham10000":
        classes_name = ["BKL", "NV", "DF", "MEL", "VASC", "BCC", "AKIEC"]
    elif args.dataset == "chestct":
        classes_name = ["Normal", "ACC", "LCC", "SCC"]
    else:
        raise print("{}, {:}".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'), colored("Not support this dataset!!!", "red")), flush=True)

    if args.checkpoint:
        print("{}, Loading the trained weights".format(
            datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')), flush=True)
        best_path = os.path.abspath(os.path.join(os.path.curdir, f"best_runs/{args.dataset}_{args.shot}/"))
        model.load_state_dict(torch.load(best_path + '/best_state.pt', weights_only=True, map_location=device))

    acc, f1, mcc, kappa, auc, cm, report = test(model, test_dataloader, args, classes_name)
    os.makedirs("confusion", exist_ok=True)
    np.save(f"confusion/confusion_{args.dataset}_{args.shot}.npy", cm)
    print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
        colored("Acc:", "blue"), acc,
        colored("F1-score:", "blue"), f1,
        colored("MCC:", "blue"), mcc,
        colored("Kappa:", "blue"), kappa,
        colored("AUC:", "blue"), auc
    ), flush=True)

    return {
        "dataset": args.dataset,
        "shot": args.shot,
        "seed": args.seed,
        "acc": round(acc, 4),
        "f1": round(f1, 4),
    }


if __name__ == "__main__":

    result = main()
    write_dataset_results(result["dataset"], [result], args)
