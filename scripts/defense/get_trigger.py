import argparse
import json
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import sys

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from utils.utils import *
import time
from torch import nn


def apply_noise_patch(
    noise, images, offset_x=0, offset_y=0, mode="change", padding=20, position="fixed"
):
    """
    noise: torch.Tensor(1, 3, pat_size, pat_size)
    images: torch.Tensor(N, 3, 512, 512)
    outputs: torch.Tensor(N, 3, 512, 512)
    """
    length = images.shape[2] - noise.shape[2]
    if position == "fixed":
        wl = offset_x
        ht = offset_y
    else:
        wl = np.random.randint(padding, length - padding)
        ht = np.random.randint(padding, length - padding)
    if images.dim() == 3:
        noise_now = noise.clone()[0, :, :, :]
        wr = length - wl
        hb = length - ht
        m = nn.ZeroPad2d((wl, wr, ht, hb))
        if mode == "change":
            images[:, ht : ht + noise.shape[2], wl : wl + noise.shape[3]] = 0
            images += m(noise_now)
        else:
            images = images + noise_now
    else:
        for i in range(images.shape[0]):
            noise_now = noise.clone()
            wr = length - wl
            hb = length - ht
            m = nn.ZeroPad2d((wl, wr, ht, hb))
            if mode == "change":
                images[
                    i : i + 1, :, ht : ht + noise.shape[2], wl : wl + noise.shape[3]
                ] = 0
                images[i : i + 1] += m(noise_now)
            else:
                images[i : i + 1] += noise_now
    return images


def narcissus_gen(
    args,
    model,
    dataset,
    source_label,
    epochs,
    l_inf_r,
    dataset_path,
    load_path,
    save_noise_path,
):
    noise_size = args.image_size
    l_inf_r = l_inf_r
    batch_size = 32
    patch_mode = "add"

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

    transform_train = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.RandomCrop(args.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    noise_normalize = transforms.Normalize(
        [0, 0, 0], std
    )  # equal to "Normalize(Image+Trigger)", to get "Trigger", not "Normalize(Trigger)"

    if dataset == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            root=dataset_path, train=True, download=False, transform=transform_train
        )
    elif dataset == "cifar100":
        ds = torchvision.datasets.CIFAR100(
            root=dataset_path, train=True, download=False, transform=transform_train
        )
    elif dataset == "tinyimagenet":
        ds = ImageFolder(
            root=os.path.join(args.data_path, "tiny-imagenet-200/train/"),
            transform=transform_train,
        )
    else:
        raise NotImplementedError("no implement")

    indices = [i for i, (_, label) in enumerate(ds) if label == source_label]  # type: ignore
    subset = Subset(ds, indices)
    dl_subset = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    noise = torch.zeros((1, 3, noise_size, noise_size), device=args.device)

    model = get_model(model, dataset, args.device)
    model.load_state_dict(torch.load(load_path))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()
    batch_pert = torch.autograd.Variable(noise.to(args.device), requires_grad=True)
    batch_opt = torch.optim.RAdam(params=[batch_pert], lr=0.01)
    sche = torch.optim.lr_scheduler.StepLR(batch_opt, 100, 0.7)
    # sche = torch.optim.lr_scheduler.CosineAnnealingLR(batch_opt, 200)

    print("Training the trigger")

    epochs_iterator = tqdm(range(epochs), desc="Narcissus Generation", ncols=100)
    epochs_iterator.set_description(
        f"\rNarcissus Generation | Epoch {0}/{epochs} | Gradient: {0} | Loss: {0}"
    )
    for minmin in epochs_iterator:
        loss_list = []
        for images, labels in dl_subset:
            images, labels = images.to(args.device), labels.to(args.device)
            new_images = torch.clone(images)
            clamp_batch_pert = torch.clamp(batch_pert, -l_inf_r, l_inf_r)
            new_images = torch.clamp(
                apply_noise_patch(
                    noise_normalize(clamp_batch_pert),
                    new_images.clone(),
                    mode=patch_mode,
                ),
                -1,
                1,
            )
            per_logits = model(new_images)
            loss = criterion(per_logits, labels)
            loss_regu = torch.mean(loss)
            batch_opt.zero_grad()
            loss_list.append(float(loss_regu.data))
            loss_regu.backward(retain_graph=True)
            batch_opt.step()
        sche.step()
        ave_loss = np.average(np.array(loss_list))
        ave_grad = np.sum(abs(batch_pert.grad).detach().cpu().numpy())  # type: ignore
        if minmin % 20 == 0:
            epochs_iterator.set_description(
                f"\rEpoch {minmin + 1}/{epochs} | Gradient: {ave_grad:.6f} | Loss: {ave_loss:.6f}"
            )
        if ave_grad == 0:
            break

    noise = torch.clamp(batch_pert, -l_inf_r, l_inf_r)
    best_noise = noise.clone().detach().cpu()
    plt.imshow(np.transpose(noise[0].detach().cpu(), (1, 2, 0)))
    plt.show()
    print("Noise max val:", noise.max())

    torch.save(best_noise, save_noise_path)

    return ave_grad, ave_loss


def main():
    parser = argparse.ArgumentParser(
        description="Parameters for calculating threshlod",
    )
    parser.add_argument(
        "--model",
        default="mobilevit",
        help="model",
        choices=[
            "lenet",
            "conv3",
            "wrn16_4",
            "res18",
            "dense121",
            "googlenet",
            "mobilenetv2",
            "mobilevit",
            "vit_b",
            "vit_s",
        ],
    )
    parser.add_argument(
        "--dataset",
        default="tinyimagenet",
        help="Dataset used to get distance(default: cifar10)",
        choices=["cifar10", "cifar100", "tinyimagenet", "imagenet10"],
    )
    parser.add_argument("--image_size", default=64, type=int, help="")
    parser.add_argument("--source_label", default=0, type=int, help="")
    parser.add_argument("--epochs", default=100, type=int, help="")
    parser.add_argument("--idx", default=1, type=int, help="")

    parser.add_argument(
        "--data_path",
        default="/usr/common/datasets/",
        help="Path to store all the relevant datasetS.",
    )
    parser.add_argument("--device", default="cuda:3", help="device to run")
    parser.add_argument("--log_dir", default="logs", help="日志保存根目录")

    args = parser.parse_args()

    idx = args.idx

    model = args.model
    dataset = args.dataset

    if args.source_label == None:
        if dataset == "cifar10":
            source_label = 9
        else:
            source_label = 0
    else:
        source_label = args.source_label

    l_inf_r = 1.0

    load_path = os.path.join(project_root, f"checkpoint/{dataset}/{model}/{idx}/clean/checkpoint.pt")
    save_dir = os.path.join(project_root, f"checkpoint/{dataset}/{model}/{idx}/clean/trigger/")
    save_path = os.path.join(project_root, f"checkpoint/{dataset}/{model}/{idx}/clean/trigger/{source_label}.pt")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(save_dir) and os.path.isdir(save_dir):
        if not os.listdir(save_dir):
            print(f"The directory '{save_dir}' is empty.")
        else:
            print(f"The directory '{save_dir}' is not empty.")
            return

    experiment_log = {
        "Parameters": {
            "dataset": dataset,
            "model": model,
            "source_label": source_label,
            "l_inf_r": l_inf_r,
            "epochs": args.epochs,
        },
        "time": "",
        "ave_grad": 0,
        "ave_loss": 0,
    }

    start_time = time.time()
    ave_grad, ave_loss = narcissus_gen(
        args,
        model,
        dataset,
        source_label=source_label,
        epochs=args.epochs,
        l_inf_r=l_inf_r,
        dataset_path=args.data_path,
        load_path=load_path,
        save_noise_path=save_path,
    )
    end_time = time.time()

    experiment_log["time"] = f"{end_time - start_time:.2f}"
    experiment_log["ave_grad"] = f"{ave_grad}"
    experiment_log["ave_loss"] = f"{ave_loss}"

    # 日志保存在checkpoint同目录下
    log_save_path = os.path.join(save_dir, f"{dataset}_{model}_{idx}.json")
    with open(log_save_path, "w") as log_file:
        json.dump(experiment_log, log_file, indent=4)

    print(experiment_log)


if __name__ == "__main__":
    main()
