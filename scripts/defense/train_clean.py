import argparse
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils.utils import *
import torch, torchvision
from utils.watermark_utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import time
from transformers import get_cosine_schedule_with_warmup


def train(args, batch_size, epochs, lr, model, dataset, save_path):

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

    if dataset == "cifar10":
        ds_train = torchvision.datasets.CIFAR10(
            args.data_path, train=True, transform=transform_train
        )
        ds_test = torchvision.datasets.CIFAR10(
            args.data_path, train=False, transform=transform_test
        )
    elif dataset == "cifar100":
        ds_train = torchvision.datasets.CIFAR100(
            args.data_path, train=True, transform=transform_train
        )
        ds_test = torchvision.datasets.CIFAR100(
            args.data_path, train=False, transform=transform_test
        )
    elif dataset == "tinyimagenet":
        ds_train = ImageFolder(
            root=os.path.join(args.data_path, "tiny-imagenet-200/train/"),
            transform=transform_train,
        )
        ds_test = TinyImageNetValDataset(
            ds_root=args.data_path, transform=transform_test
        )

    elif dataset == "imagenet10":
        ds_train = ImageNet10Dataset(
            ds_root=args.data_path, train=True, transform=transform_train
        )
        ds_test = ImageNet10Dataset(
            ds_root=datasets_root, train=False, transform=transform_test
        )

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4)

    model = get_model(model, dataset, device=args.device)

    if args.model in [
        "res18",
        "dense121",
        "mobilenetv2",
        "googlenet",
        "wrn16_4",
        "mobilevit",
    ]:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    if args.model in ["vit_b", "vit_s"]:
        num_training_steps = epochs * len(dl_train)
        num_warmup_steps = int(num_training_steps * 0.1)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    criterion = torch.nn.CrossEntropyLoss()

    epoch_iterator = tqdm(range(epochs), desc="Train Clean", ncols=120)
    epoch_iterator.set_description(f"\rTrain Clean | Epoch {0}/{epochs} | ACC: {0}")
    for epoch in epoch_iterator:
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
        epoch_iterator.set_description(
            f"\rTrain Clean Epoch {epoch + 1}/{epochs} | ACC: {acc}"
        )

    acc = round(test(model, dl_test, args.device), 4)
    print("ACC:", acc)
    torch.save(model.state_dict(), save_path)

    return acc


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
            "imagenet10",
        ],
    )
    parser.add_argument("--image_size", default=64, type=int, help="Image size")

    parser.add_argument("--epochs", default=20, type=int, help="")
    parser.add_argument("--lr", default=1e-1, type=float, help="")
    parser.add_argument("--batch_size", default=256, type=int, help="")

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

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    model = args.model
    dataset = args.dataset

    # 使用项目根目录下的 checkpoint 文件夹
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_dir = os.path.join(project_root, f"checkpoint/{dataset}/{model}/{idx}/clean/")
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
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
        },
    }

    start_time = time.time()
    result = train(args, batch_size, epochs, lr, model, dataset, save_path=save_path)
    end_time = time.time()
    experiment_log["accuracy"] = result
    experiment_log["time"] = f"{end_time - start_time:.2f}"

    # 日志保存在checkpoint同目录下
    log_save_path = os.path.join(save_dir, f"{dataset}_{model}_{idx}.json")
    with open(log_save_path, "w") as log_file:
        json.dump(experiment_log, log_file, indent=4)

    print(experiment_log)


if __name__ == "__main__":
    main()
