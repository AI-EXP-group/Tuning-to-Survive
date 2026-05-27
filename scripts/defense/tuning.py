import argparse
import json
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)


import torchvision
from utils.utils import *
from collections import OrderedDict
from functional_model.functional_wrn import *
from functional_model.functional_res18 import *
from functional_model.functional_dense121 import *
from functional_model.functional_googlenet import *
from functional_model.functional_mobilevit import (
    functional_mobilevit_s,
)
import torch
from utils.watermark_utils import *
from tqdm import tqdm
import time
from torch.utils.data import DataLoader, Subset


def train(
    args,
    model,
    dataset,
    mode,
    epochs,
    lr_outer,
    lr_inner,
    alpha,
    t_load_path,
    s_load_path,
    save_path,
    idx=0,
    source_label1=0,
    source_label2=1,
    target_label=2,
    trigger=None,
):
    if mode == "feature":
        if trigger != None:
            trigger = trigger
        else:
            trigger = torch.load(
                f"checkpoint/{dataset}/{model}/{idx}/clean/trigger/{source_label1}.pt"
            ).squeeze(0)
    elif mode == "random_trigegr":
        trigger = torch.load("trigger/random_0_0.7.pth").squeeze(0)
        mode = "feature"
    else:
        trigger = None

    if dataset == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif dataset == "tinyimagenet":
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2302, 0.2265, 0.2262]
    else:
        raise NotImplementedError("no implement")

    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.RandomCrop(args.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    if dataset == "cifar10":
        ds_distill = torchvision.datasets.CIFAR10(
            args.data_path, train=True, transform=transform
        )
        # indices = np.random.choice(len(ds_distill), 1000, replace=False)
        # ds_distill = Subset(ds_distill, indices)  # type: ignore
        ds_test = torchvision.datasets.CIFAR10(
            args.data_path, train=False, transform=transform_test
        )
    elif dataset == "cifar100":
        ds_distill = torchvision.datasets.CIFAR100(
            args.data_path, train=True, transform=transform
        )
        ds_test = torchvision.datasets.CIFAR100(
            args.data_path, train=False, transform=transform_test
        )
    elif dataset == "tinyimagenet":
        ds_distill = ImageFolder(
            root=os.path.join(args.data_path, "tiny-imagenet-200/train/"),
            transform=transform,
        )
        ds_test = TinyImageNetValDataset(
            ds_root=args.data_path, transform=transform_test
        )
    len_ds_distill = len(ds_distill)

    dl_distill = DataLoader(
        ds_distill, batch_size=args.inner_batch_size, shuffle=True, num_workers=4
    )

    dl_test = DataLoader(ds_test, batch_size=16, shuffle=False, num_workers=4)
    ds_watermark = get_watermark_ds(
        mode,
        args=args,
        model=model,
        dataset=dataset,
        trigger=trigger,
        source_label1=source_label1,
        source_label2=source_label2,
        target_label=target_label,
        num=500,
    )
    dl_watermark = DataLoader(
        ds_watermark, batch_size=args.outer_batch_size, shuffle=False, num_workers=4
    )

    model_t = get_model(model, dataset, args.device)
    model_s = get_model(model, dataset, args.device)
    model_t.load_state_dict(torch.load(t_load_path, map_location=args.device))
    model_s.load_state_dict(torch.load(s_load_path, map_location=args.device))

    # t_acc = test(model_t, dl_test, args.device)
    # t_wsr = test(model_t, dl_watermark, args.device)
    # s_acc = test(model_s, dl_test, args.device)
    # s_wsr = test(model_s, dl_watermark, args.device)
    # print(
    #     f"Initial Acc: T-Acc: {t_acc:.4f}, T-WSR: {t_wsr:.4f}, S-Acc: {s_acc:.4f}, S-WSR: {s_wsr:.4f}\n"
    # )

    if model == "res18":
        functional_model = functional_res18
    elif model == "dense121":
        functional_model = functional_dense121
    elif model == "googlenet":
        functional_model = functional_googlenet
    elif model == "wrn":
        functional_model = functional_wrn
    elif model == "mobilevit":
        functional_model = functional_mobilevit_s

    criterion_distill = loss_fn_kd
    criterion_quiz = torch.nn.CrossEntropyLoss()
    optimizer_t = torch.optim.SGD(model_t.parameters(), lr=lr_outer)
    lr_scheduler_t = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_t,
        T_max=int(len_ds_distill / args.inner_batch_size * epochs),
    )

    print(
        f"{'Step':<6}| {'T-Acc':<7}| {'T-WSR':<7}| {'S-Acc':<7}| {'S-WSR':<7}| {'L-D':<7}| {'L-W':<7}"
    )

    for epoch in range(epochs):
        # ------------------------------------------------train teacher------------------------------------------------
        epoch_iterator = tqdm(dl_distill, ncols=100)
        # lr = lr_inner

        for d_step, d_batch in enumerate(
            epoch_iterator, start=0
        ):  # len_ds_distill / dl_distill.inner_batch_size
            # ----------------------------------------distll student----------------------------------------
            weight_s, weight_s_nograd = get_weights(model_s)
            model_s.train()
            model_t.eval()
            loss_normal = 0
            num_batch = 0
            batch_size = args.inner_distill_batch_size
            num_iterations = (
                len(d_batch[0]) // batch_size
            )  # dl_distill.batch_size // batch_size (cifar10:100//10=10)
            for i in range(num_iterations):
                start = i * batch_size
                end = start + batch_size
                input = d_batch[0][start:end].to(args.device)
                label = d_batch[1][start:end].to(args.device)
                tea_out = model_t(input)
                stu_out = (
                    model_s(input)
                    if i == 0
                    else functional_model(weight_s, weight_s_nograd, input)
                )
                hard = tea_out.data.max(1)[1]
                loss_distill = criterion_distill(
                    stu_out, hard, tea_out, alpha=1, T=1  # type: ignore
                )
                loss_normal += criterion_quiz(tea_out, label)

                grads = torch.autograd.grad(
                    loss_distill,
                    model_s.parameters() if i == 0 else weight_s.values(),  # type: ignore
                    create_graph=True,
                    retain_graph=True,
                )
                weight_s = OrderedDict(
                    (name, param - lr_inner * grad)
                    for ((name, param), grad) in zip(weight_s.items(), grads)
                )

                num_batch += 1
            loss_normal /= num_batch

            # --------------------------------train teacher using second derivatives--------------------------------
            model_t.train()
            loss_watermark = 0
            num_batch = 0
            for step, batch in enumerate(dl_watermark):
                watermark, target_label = batch[0].to(args.device), batch[1].to(
                    args.device
                )
                output_watermark = functional_model(
                    weight_s, weight_s_nograd, watermark
                )
                loss_watermark += criterion_quiz(output_watermark, target_label)
                num_batch += 1

            loss_watermark /= num_batch
            t_grads = torch.autograd.grad(
                loss_watermark + alpha * loss_normal, model_t.parameters()  # type: ignore
            )

            for p, gr in zip(model_t.parameters(), t_grads):
                p.grad = gr
            torch.nn.utils.clip_grad_norm_(model_t.parameters(), 1)
            optimizer_t.step()

            for p in model_t.parameters():
                p.grad = None

            for p in model_s.parameters():
                p.grad = None

            del t_grads
            del grads

            lr_scheduler_t.step()

            # ------------------------------------------------ test ------------------------------------------------
            if (
                d_step % int((len_ds_distill / args.inner_batch_size) / 10) == 0
                or d_step == len(epoch_iterator) - 1
            ):
                load_weights(model_s, weight_s, weight_s_nograd)
                t_acc = test(model_t, dl_test, args.device)
                t_wsr = test(model_t, dl_watermark, args.device)
                s_acc = test(model_s, dl_test, args.device)
                s_wsr = test(model_s, dl_watermark, args.device)
                # epoch_iterator.set_description(
                #     f"\r{d_step}\t|{t_acc:.4f}\t|{t_wsr:.4f}\t|{s_acc:.4f}\t|{s_wsr:.4f}|\t{loss_normal.item():.4f}\t|{loss_watermark.item():.4f}"  # type: ignore
                # )
                desc = (
                    f"{d_step:<6}| {t_acc:<7.4f}| {t_wsr:<7.4f}| "
                    f"{s_acc:<7.4f}| {s_wsr:<7.4f}| {loss_normal.item():<7.4f}| {loss_watermark.item():<7.4f}"  # type: ignore
                )
                epoch_iterator.set_description(desc)

    torch.save(model_t.state_dict(), save_path)
    return t_acc, t_wsr, s_acc, s_wsr


def main():
    parser = argparse.ArgumentParser(
        description="",
    )
    parser.add_argument(
        "--model",
        default="mobilevit",
        help="model",
        choices=[
            "res18",
            "wrn16_4",
            "dense121",
            "googlenet",
            "mobilevit",
        ],
    )
    parser.add_argument(
        "--dataset",
        default="tinyimagenet",
        help="Dataset used to get distance(default: cifar10)",
        choices=[
            "cifar10",
            "cifar100",
            "tinyimagenet",
        ],
    )
    parser.add_argument("--image_size", default=64, type=int, help="")
    parser.add_argument("--idx", default=1, type=int, help="")
    parser.add_argument("--source_label1", default=None, type=int, help="")
    parser.add_argument("--source_label2", default=0, type=int, help="")
    parser.add_argument("--target_label", default=1, type=int, help="")
    parser.add_argument("--mode", default="feature", type=str, help="")
    parser.add_argument(
        "--alpha", default=15, type=int, help=""
    )  # cifar10:50 cifar100:15 #dense121:30 mobilevit:15

    parser.add_argument("--lr_outer", default=0.005, type=float, help="")  # res18 0.005
    parser.add_argument(
        "--lr_inner",
        default=0.001,
        type=float,
        help="0.0001 for mobilevit, 0.001 for others",
    )  # 0.005 - 0.01 bigger batch_size -> bigger lr
    parser.add_argument("--epochs", default=5, type=int, help="")
    parser.add_argument("--inner_batch_size", default=100, type=int, help="")
    parser.add_argument("--inner_distill_batch_size", default=10, type=int, help="")
    parser.add_argument("--outer_batch_size", default=500, type=int, help="")

    parser.add_argument(
        "--data_path",
        default="/usr/common/datasets/",
        help="Path to store all the relevant datasetS.",
    )
    parser.add_argument("--device", default="cuda:3", help="device to run")

    args = parser.parse_args()

    lr_outer = args.lr_outer
    lr_inner = args.lr_inner
    model = args.model
    mode = args.mode
    dataset = args.dataset
    # if args.source_label1 == None:
    #     if dataset == "cifar10":
    #         source_label1, source_label2, target_label = 9, 1, 6
    #     elif dataset == "cifar100":
    #         source_label1, source_label2, target_label = 0, 1, 96
    #     elif dataset == "tinyimagenet":
    #         source_label1, source_label2, target_label = 0, 1, 2
    # else:
    #     source_label1, source_label2, target_label = (
    #         args.source_label1,
    #         args.source_label2,
    #         args.target_label,
    #     )
    if dataset == "cifar10":
        source_label1 = args.source_label1 if args.source_label1 else 9
        source_label2 = args.source_label2 if args.source_label2 else 1
        target_label = args.target_label if args.target_label else 6
    elif dataset == "cifar100":
        source_label1 = args.source_label1 if args.source_label1 else 0
        source_label2 = args.source_label2 if args.source_label2 else 1
        target_label = args.target_label if args.target_label else 96
    elif dataset == "tinyimagenet":
        source_label1 = args.source_label1 if args.source_label1 else 0
        source_label2 = args.source_label2 if args.source_label2 else 1
        target_label = args.target_label if args.target_label else 2

    epochs = args.epochs
    alpha = args.alpha

    idx = args.idx

    t_load_path = f"checkpoint/{dataset}/{model}/{idx}/watermarked/{mode}/checkpoint.pt"
    s_load_path = f"checkpoint/{dataset}/{model}/{idx}/clean/checkpoint.pt"
    save_dir = f"checkpoint/{dataset}/{model}/{idx}/t2s/{mode}/"
    save_path = save_dir + "checkpoint.pt"

    if os.path.exists(save_dir) and os.path.isdir(save_dir):
        if not os.listdir(save_dir):
            print(f"The directory '{save_dir}' is empty.")
        else:
            print(f"The directory '{save_dir}' is not empty.")
            return
    else:
        os.makedirs(save_dir)

    experiment_log = {
        "Experiment Log": [],
    }

    experiment_log = {
        "Experiment Index": idx,
        "Parameters": {
            "dataset": dataset,
            "model": model,
            "watermark_mode": mode,
            "source_label1": source_label1,
            "source_label2": source_label2,
            "target_label": target_label,
            "epochs": epochs,
            "alpha": alpha,
            "lr_outer": lr_outer,
            "lr_inner": lr_inner,
            "inner_batch_size": args.inner_batch_size,
            "inner_distill_batch_size": args.inner_distill_batch_size,
            "outer_batch_size": args.outer_batch_size,
        },
    }

    start_time = time.time()
    result = train(
        args,
        model,
        dataset,
        mode,
        epochs,
        lr_outer,
        lr_inner,
        alpha=alpha,
        t_load_path=t_load_path,
        s_load_path=s_load_path,
        save_path=save_path,
        idx=idx,
        source_label1=source_label1,
        source_label2=source_label2,
        target_label=target_label,
    )
    end_time = time.time()
    experiment_log["time"] = f"{end_time - start_time:.2f}"
    experiment_log["target_acc"] = result[0]
    experiment_log["target_wsr"] = result[1]
    experiment_log["stolen_acc"] = result[2]
    experiment_log["stolen_wsr"] = result[3]

    with open(save_dir + "experiment_log.json", "w") as log_file:
        json.dump(experiment_log, log_file, indent=4)

    print(experiment_log)


if __name__ == "__main__":
    main()
    # hyperparameter_tuning()


# lr_inner = 0.001, lr_outer = 0.005, alpha =40, epochs = 1
# distill_batch_size = 100, inner_distill_batch_size = 5,   89.39,  99.40
# distill_batch_size = 100, inner_distill_batch_size = 10,  88.30,  99.80 √     # alpha = 50    89.80， 98.60
# distill_batch_size = 100, inner_distill_batch_size = 20,  90.18,  13.00

# lr_inner = 0.005, , lr_outer = 0.01, alpha = 40, epochs = 1
# distill_batch_size = 100, inner_distill_batch_size = 5,   ×
# distill_batch_size = 100, inner_distill_batch_size = 10,  91.20,  0
# distill_batch_size = 100, inner_distill_batch_size = 20,  89.96,  20.20

# lr_inner = 0.001, lr_outer = 0.001, alpha =40, epochs = 1
# distill_batch_size = 100, inner_distill_batch_size = 5,   89.12   1.80
# distill_batch_size = 100, inner_distill_batch_size = 10,  89.12   74.40
# distill_batch_size = 100, inner_distill_batch_size = 20,  90.28   34.80
