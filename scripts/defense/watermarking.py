import argparse
import json
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from utils.utils import *
import torch, torchvision
from utils.watermark_utils import *
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import time


def watermarking(
    args,
    batch_size,
    epochs,
    lr,
    model,
    dataset,
    mode,
    load_path,
    save_path,
    num,
    source_label1=0,
    source_label2=1,
    target_label=2,
    trigger=None,
    idx=0,
):

    transform_train = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.RandomCrop(args.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            get_transform(dataset),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            get_transform(dataset),
        ]
    )

    if mode == "feature":
        if trigger != None:
            trigger = trigger
        else:
            trigger = torch.load(
                f"checkpoint/{dataset}/{model}/{idx}/clean/trigger/{source_label1}.pt"
            ).squeeze(0)
    else:
        trigger = None

    if dataset == "cifar10":
        ds_normal = torchvision.datasets.CIFAR10(
            args.data_path, train=True, transform=transform_train
        )
        ds_test = torchvision.datasets.CIFAR10(
            args.data_path, train=False, transform=transform_test
        )
    elif dataset == "cifar100":
        ds_normal = torchvision.datasets.CIFAR100(
            args.data_path, train=True, transform=transform_train
        )
        ds_test = torchvision.datasets.CIFAR100(
            args.data_path, train=False, transform=transform_test
        )
    elif dataset == "tinyimagenet":
        ds_normal = ImageFolder(
            root=os.path.join(args.data_path, "tiny-imagenet-200/train/"),
            transform=transform_train,
        )
        ds_test = TinyImageNetValDataset(
            ds_root=args.data_path, transform=transform_test
        )

    elif dataset == "imagenet10":
        ds_normal = ImageNet10Dataset(
            ds_root=args.data_path, train=True, transform=transform_train
        )
        ds_test = ImageNet10Dataset(
            ds_root=datasets_root, train=False, transform=transform_test
        )

    ds_watermark = get_watermark_ds(
        mode,
        args=args,
        model=model,
        dataset=dataset,
        num=num,
        trigger=trigger,
        source_label1=source_label1,
        source_label2=source_label2,
        target_label=target_label,
    )
    if mode == "mix":
        ds_normal_mix = get_watermark_ds(
            "normal_mix",
            args=args,
            model=model,
            dataset=dataset,
            num=500,
            source_label1=source_label1,
            source_label2=source_label2,
            target_label=target_label,
        )
        ds_train = ConcatDataset([ds_watermark, ds_normal_mix, ds_normal])
    else:
        ds_train = ConcatDataset([ds_watermark, ds_normal])
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4)
    dl_watermark_test = DataLoader(
        get_watermark_ds(
            mode,
            args=args,
            model=model,
            dataset=dataset,
            num=num,
            trigger=trigger,
            target_label=target_label,
        ),
        batch_size=500,
        shuffle=False,
        num_workers=4,
    )

    model = get_model(model, dataset, device=args.device)
    model.load_state_dict(torch.load(load_path))
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=[5, 15], gamma=0.5
    )

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), ncols=100):
        model.train()
        for batch in dl_train:
            input, target = batch[0].to(args.device), batch[1].to(args.device)
            output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

    acc = round(test(model, dl_test, args.device), 4)
    wsr = round(test(model, dl_watermark_test, args.device), 4)

    torch.save(model.state_dict(), save_path)

    return acc, wsr


def main():
    parser = argparse.ArgumentParser(description="")
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
            "vit_b",
            "vit_s",
        ],
    )
    parser.add_argument(
        "--dataset",
        default="tinyimagenet",
        type=str,
        choices=["cifar10", "cifar100", "tinyimagenet"],
    )
    parser.add_argument("--image_size", default=64, type=int, help="")
    parser.add_argument("--source_label1", default=None, type=int, help="")
    parser.add_argument("--source_label2", default=0, type=int, help="")
    parser.add_argument("--target_label", default=1, type=int, help="")
    parser.add_argument("--idx", default=1, type=int, help="")

    parser.add_argument("--mode", default="feature", type=str, help="")
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="3e-3 for mobilevit, 1e-2 for others"
    )
    parser.add_argument("--epochs", default=5, type=int, help="")
    parser.add_argument("--batch_size", default=128, type=int, help="")

    parser.add_argument(
        "--data_path",
        default="/usr/common/datasets/",
        help="Path to store all the relevant datasetS.",
    )
    parser.add_argument("--device", default="cuda:3", help="device to run")

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    model = args.model
    dataset = args.dataset

    watermark_mode = args.mode

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

    idx = args.idx

    load_path = f"checkpoint/{dataset}/{model}/{idx}/clean/checkpoint.pt"
    save_dir = f"checkpoint/{dataset}/{model}/{idx}/watermarked/{watermark_mode}/"
    save_path = (
        f"checkpoint/{dataset}/{model}/{idx}/watermarked/{watermark_mode}/"
        + "checkpoint.pt"
    )
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
            "watermark_mode": watermark_mode,
            "source_label1": source_label1,
            "source_label2": source_label2,
            "target_label": target_label,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
        },
    }

    start_time = time.time()
    result = watermarking(
        args,
        batch_size,
        epochs,
        lr,
        model,
        dataset,
        watermark_mode,
        load_path=load_path,
        save_path=save_path,
        num=500,
        source_label1=source_label1,
        source_label2=source_label2,
        target_label=target_label,
        idx=idx,
    )
    end_time = time.time()
    experiment_log["time"] = f"{end_time - start_time:.2f}"
    experiment_log["acc"] = result[0]
    experiment_log["wsr"] = result[1]

    with open(save_dir + "experiment_log.json", "w") as log_file:
        json.dump(experiment_log, log_file, indent=4)

    print(experiment_log)


if __name__ == "__main__":

    main()
    # temp()
