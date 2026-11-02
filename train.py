import math
import os
from tqdm import tqdm
from termcolor import colored
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from test import test


def print_branch_weights(model):
    branch_weights = []
    for module in model.modules():
        if all(hasattr(module, name) for name in ("weight_hca", "weight_ssa", "weight_ffed")):
            hca = module.weight_hca.detach().item()
            ssa = module.weight_ssa.detach().item()
            ffed = module.weight_ffed.detach().item()
            branch_weights.append((hca, ssa, ffed))

    if not branch_weights:
        print("No branch fusion weights found.", flush=True)
        return

    print("Trained branch fusion weights:", flush=True)
    for idx, (hca, ssa, ffed) in enumerate(branch_weights):
        print(
            f"  AdaptiveScaleSelection {idx}: "
            f"HCA={hca:.6f}, SSA={ssa:.6f}, FFED={ffed:.6f}",
            flush=True
        )

    avg_hca = sum(weight[0] for weight in branch_weights) / len(branch_weights)
    avg_ssa = sum(weight[1] for weight in branch_weights) / len(branch_weights)
    avg_ffed = sum(weight[2] for weight in branch_weights) / len(branch_weights)
    print(
        f"  Average: HCA={avg_hca:.6f}, SSA={avg_ssa:.6f}, FFED={avg_ffed:.6f}",
        flush=True
    )


def print_global_fusion_weights(model):
    if not all(hasattr(model, name) for name in ("alpha", "beta", "gamme")):
        print("No global fusion weights found.", flush=True)
        return

    print(
        "Trained global fusion weights: "
        f"alpha={model.alpha.detach().item():.6f}, "
        f"beta={model.beta.detach().item():.6f}, "
        f"gamme={model.gamme.detach().item():.6f}",
        flush=True
    )


def print_branch_activation_stats(model, dataloader):
    modules = [
        module for module in model.modules()
        if all(hasattr(module, name) for name in ("record_branch_stats", "branch_stats"))
    ]
    if not modules:
        print("No branch activation stats modules found.", flush=True)
        return

    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training

    stats = [
        {
            "count": 0,
            "multi_scale": 0.0,
            "hca": 0.0,
            "ssa": 0.0,
            "ffed": 0.0,
            "weighted_hca": 0.0,
            "weighted_ssa": 0.0,
            "weighted_ffed": 0.0,
            "final": 0.0,
        }
        for _ in modules
    ]

    try:
        for module in modules:
            module.record_branch_stats = True
            module.branch_stats = None

        model.eval()
        with torch.no_grad():
            for X, _ in dataloader:
                model(X)
                for idx, module in enumerate(modules):
                    current = module.branch_stats
                    if current is None:
                        continue
                    stats[idx]["count"] += 1
                    for key, value in current.items():
                        stats[idx][key] += value
    finally:
        for module in modules:
            module.record_branch_stats = False
        if was_training:
            model.train()
        else:
            model.eval()
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    keys = [
        "multi_scale",
        "hca",
        "ssa",
        "ffed",
        "weighted_hca",
        "weighted_ssa",
        "weighted_ffed",
        "final",
    ]
    print("Branch activation L2 norms on validation set:", flush=True)
    averaged = []
    for idx, item in enumerate(stats):
        if item["count"] == 0:
            continue
        avg = {key: item[key] / item["count"] for key in keys}
        averaged.append(avg)
        print(
            f"  AdaptiveScaleSelection {idx}: "
            f"MS={avg['multi_scale']:.6f}, "
            f"HCA={avg['hca']:.6f}, SSA={avg['ssa']:.6f}, FFED={avg['ffed']:.6f}, "
            f"wHCA={avg['weighted_hca']:.6f}, wSSA={avg['weighted_ssa']:.6f}, "
            f"wFFED={avg['weighted_ffed']:.6f}, Final={avg['final']:.6f}",
            flush=True
        )

    if averaged:
        overall = {
            key: sum(item[key] for item in averaged) / len(averaged)
            for key in keys
        }
        print(
            "  Average: "
            f"MS={overall['multi_scale']:.6f}, "
            f"HCA={overall['hca']:.6f}, SSA={overall['ssa']:.6f}, FFED={overall['ffed']:.6f}, "
            f"wHCA={overall['weighted_hca']:.6f}, wSSA={overall['weighted_ssa']:.6f}, "
            f"wFFED={overall['weighted_ffed']:.6f}, Final={overall['final']:.6f}",
            flush=True
        )


def trainer(model, train_dataloader, val_dataloader, args):
    print("{}, Start to train".format(
        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')), flush=True)

    best_run_name = getattr(args, "best_run_name", f"{args.dataset}_{args.shot}")
    best_path = os.path.abspath(os.path.join(os.path.curdir, "best_runs", best_run_name))
    if not os.path.exists(best_path):
        os.makedirs(best_path)

    loss_fn = nn.CrossEntropyLoss()
    no_improve_num = 0
    print("{}, Training main network".format(
        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')), flush=True)
    print(
        "{}, Using {} KL".format(
            datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
            "class-conditional prior" if args.use_class_prior_kl else "standard normal prior"
        ),
        flush=True
    )

    best_f1 = 0
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999), eps=0.000001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )
    for ep in range(args.epochs):
        train_dataloader = tqdm(train_dataloader, total=len(train_dataloader), ncols=80, leave=False,
                                desc=colored('Training on Train', 'yellow'))
        model.train()
        izy_bound = 0.
        izx_bound = 0.
        for i, batch in enumerate(train_dataloader):
            X, y = batch
            logits, (x_mu, x_std) = model(X)
            class_loss = loss_fn(logits, y)
            if args.use_class_prior_kl:
                prior_mu, prior_std = model.get_class_prior(y)
                info_loss = 0.5 * (
                    2 * (torch.log(prior_std + 1e-8) - torch.log(x_std + 1e-8))
                    + ((x_mu - prior_mu) / (prior_std + 1e-8)) ** 2
                    + (x_std / (prior_std + 1e-8)) ** 2
                    - 1
                )
            else:
                mu2 = torch.zeros_like(x_mu)
                std2 = torch.ones_like(x_std)
                info_loss = 0.5 * (
                    2 * (torch.log(std2 + 1e-8) - torch.log(x_std + 1e-8))
                    + ((x_mu - mu2) / (std2 + 1e-8)) ** 2
                    + (x_std / (std2 + 1e-8)) ** 2
                    - 1
                )
            info_loss = info_loss.mean()
            ffed_loss = model.get_ffed_loss(y, args.use_class_prior_kl)
            total_loss = class_loss + args.beta * info_loss + 0.01 * ffed_loss

            izy_bound += (math.log(10, 2) - class_loss).item()
            izx_bound += info_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


        scheduler.step()

        acc, f1, mcc, kappa, auc, _, _ = test(model, val_dataloader, args)

        print(
            "{}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
                colored("Acc:", "blue"), acc,
                colored("F1-score:", "blue"), f1,
                colored("MCC:", "blue"), mcc,
                colored("Kappa:", "blue"), kappa,
                colored("AUC:", "blue"), auc,
                colored("IZY:", "blue"), izy_bound / len(train_dataloader),
                colored("IZX:", "blue"), izx_bound / len(train_dataloader),
            ), flush=True)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path + '/best_state.pt')
            print("{}, {}".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
                                  colored("saving the best model state", "red")), flush=True)
            no_improve_num = 0
        else:
            no_improve_num += 1

        if no_improve_num >= args.patience:
            print("{}, Early stopping -- no improving".format(
                datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')), flush=True)
            break

    print("{}, End of training. Restore the best weights".format(
        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')), flush=True)
    model.load_state_dict(torch.load(best_path + '/best_state.pt', weights_only=True))
    print_global_fusion_weights(model)
    print_branch_weights(model)
    print_branch_activation_stats(model, val_dataloader)
