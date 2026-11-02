import datetime

import torch
import torch.nn.functional as F
import numpy as np
import os
from termcolor import colored
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

class ImageDataset(Dataset):

    def __init__(self, data, label, label2id, device):

        self.data = data
        self.label = []
        for la in label:
            self.label.append(label2id[la])
        self.device = device

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        sample = self.data[item]
        category = self.label[item]
        return torch.tensor(sample, dtype=torch.float32, device=self.device), torch.tensor(category, dtype=torch.long, device=self.device)


def _load_data(args):
    data_root = _resolve_data_root(args)

    train_data = np.load(os.path.join(data_root, f"token_{args.dataset}", f"train_{args.shot}_shot.token.npy"))
    train_label = np.load(
        os.path.join(data_root, f"token_{args.dataset}", f"train_{args.shot}_shot.label.npy"),
        allow_pickle=True,
    )

    val_data = np.load(os.path.join(data_root, f"token_{args.dataset}", "val_10_shot.token.npy"))
    val_label = np.load(
        os.path.join(data_root, f"token_{args.dataset}", "val_10_shot.label.npy"),
        allow_pickle=True,
    )

    test_data = np.load(os.path.join(data_root, f"token_{args.dataset}", "test.token.npy"))
    test_label = np.load(
        os.path.join(data_root, f"token_{args.dataset}", "test.label.npy"),
        allow_pickle=True,
    )

    return train_data, train_label, val_data, val_label, test_data, test_label


def _resolve_data_root(args):
    explicit_root = getattr(args, "data_root", None)
    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []

    if explicit_root:
        candidates.append(explicit_root)

    candidates.extend([
        os.path.abspath(os.path.join(module_dir, "..", "classification_tokens")),
        os.path.abspath(os.path.join(module_dir, "classification_tokens")),
        os.path.abspath(os.path.join(module_dir, "..", "MoE_IB", "classification_tokens")),
    ])

    for candidate in candidates:
        dataset_dir = os.path.join(candidate, f"token_{args.dataset}")
        if os.path.exists(dataset_dir):
            return candidate

    raise FileNotFoundError(
        "classification_tokens not found. Tried: "
        + ", ".join(os.path.abspath(candidate) for candidate in candidates)
    )


def _get_loader(args, device):

    if args.dataset == "esophageal":
        label2id = {"SG_Cancer": 0, "SG_Oesophagitis": 1, "SG_HGME": 2, "SG_Esophageal_Varices": 3, "SG_Esophageal_Submucosal_Bulge": 4}
    elif args.dataset == "gastric":
        label2id = {"W_Cancer": 0, "W_Ulcer": 1, "W_Polyps": 2, "W_Submucosal_Tumors": 3}
    elif args.dataset == "ham10000":
        label2id = {"bkl": 0, "nv": 1, "df": 2, "mel": 3, "vasc": 4, "bcc": 5, "akiec": 6}
    elif args.dataset == "chestct":
        label2id = {"normal": 0, "Adenocarcinoma": 1, "large-cell-carcinoma": 2, "squamous-cell-carcinoma": 3}
    else:
        raise print("{}, {:}".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'), colored("Not support this dataset!!!", "red")), flush=True)

    id2label = {v: k for k, v in label2id.items()}
    num_class = len(label2id)

    print("{}, Loading dataset for {}".format(datetime.datetime.now(), args.mode))
    print("{}, Data root: {}".format(datetime.datetime.now(), _resolve_data_root(args)))
    train_data, train_label, val_data, val_label, test_data, test_label = _load_data(args)
    print("{}, #train {}, #val {}, #test {}".format(datetime.datetime.now(), len(train_label), len(val_label), len(test_label)))

    train_dataset = ImageDataset(train_data, train_label, label2id, device)
    val_dataset = ImageDataset(val_data, val_label, label2id, device)
    test_dataset = ImageDataset(test_data, test_label, label2id, device)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, id2label, num_class
