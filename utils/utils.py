import json
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    ImageFolder,
    STL10,
)
import numpy as np
import random

from networks import (
    mobilevit,
    resnet,
    wresnet,
    densenet,
    googlenet,
    mobilenetv2,
    vit,
    vit_s,
)

datasets_root = "/usr/common/datasets/"


def get_dataloader(
    dataset,
    train=True,
    batch_size=64,
    shuffle=True,
    drop_last=False,
    dataset_ID=None,
    transform=True,
):
    if dataset_ID == None:
        dataset_ID = dataset
    transform = get_transform(dataset_ID) if transform else None

    if dataset.lower() == "cifar10":
        data = CIFAR10(datasets_root, train=train, transform=transform)

    elif dataset.lower() == "cifar100":
        data = CIFAR100(datasets_root, train=train, transform=transform)

    elif dataset.lower() == "tinyimagenet":
        if train:
            data = ImageFolder(
                root=f"{datasets_root}tiny-imagenet-200/train/images/",
                transform=transform,
            )
        else:
            data = TinyImageNetValDataset(ds_root=datasets_root, transform=transform)

    elif dataset.lower() == "stl10":
        data = STL10(datasets_root, split="train+unlabeled", transform=transform)

    elif dataset.lower() == "imagenet10":
        data = ImageNet10Dataset(
            ds_root=datasets_root, train=train, transform=transform
        )

    else:
        raise NotImplementedError("no implement")

    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
        num_workers=8,
    )


class TinyImageNetValDataset(torch.utils.data.Dataset):
    def __init__(self, ds_root=datasets_root, transform=None):
        self.val_dir = os.path.join(ds_root, "tiny-imagenet-200/val/images")
        annotations_file = os.path.join(
            ds_root, "tiny-imagenet-200/val/val_annotations.txt"
        )
        self.transform = transform
        self.image_labels = []

        # 读取 val_annotations.txt
        with open(annotations_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split("\t")
                filename, label = parts[0], parts[1]
                self.image_labels.append((filename, label))

        # 构造 label 到 index 的映射
        self.label_to_idx = {
            label: idx
            for idx, label in enumerate(sorted(set(l for _, l in self.image_labels)))
        }

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        filename, label = self.image_labels[idx]
        img_path = os.path.join(self.val_dir, filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_idx = self.label_to_idx[label]
        return image, label_idx


class ImageNet10Dataset(torch.utils.data.Dataset):
    def __init__(self, ds_root=datasets_root, train=True, transform=None):
        self.transform = transform
        dataset = ImageFolder(os.path.join(ds_root, "imagenet-10"), transform=transform)
        split_file = "split_indices.json"

        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                indices = json.load(f)
            train_indices = indices["train"]
            test_indices = indices["test"]
        else:
            num_samples = len(dataset)
            indices = torch.randperm(num_samples).tolist()
            train_size = int(0.8 * num_samples)
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]

            with open(split_file, "w") as f:
                json.dump({"train": train_indices, "test": test_indices}, f)

        self.dataset = torch.utils.data.Subset(
            dataset, train_indices if train else test_indices
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_transform(dataset):
    if dataset.lower() == "cifar10":
        normalize = transforms.Normalize(
            [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        )
    elif dataset.lower() == "cifar100":
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )
    elif dataset.lower() == "tinyimagenet":
        normalize = transforms.Normalize(
            (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
        )
    elif dataset.lower() == "imagenet10":
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        raise NotImplementedError("no implement")

    return transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )


def get_model(arch: str, dataset: str, device="cuda"):

    if dataset.lower() == "cifar10":
        image_size = 32
        num_classes = 10
    elif dataset.lower() == "cifar100":
        image_size = 32
        num_classes = 100
    elif dataset.lower() == "tinyimagenet":
        if arch == "vit_b":
            image_size = 224
        else:
            image_size = 64
        num_classes = 200
    elif dataset.lower() == "imagenet10":
        image_size = 224
        num_classes = 10
    else:
        raise NotImplementedError("no implement")

    if arch == "wrn":
        model = wresnet.wrn_16_4(num_classes=num_classes)

    elif arch == "res18":
        model = resnet.resnet18(num_classes=num_classes)

    elif arch == "dense121":
        model = densenet.densenet_cifar(num_classes=num_classes)

    elif arch == "googlenet":
        model = googlenet.GoogleNet(num_class=num_classes)

    elif arch == "mobilenetv2":
        raise NotImplementedError("MobileNetV2 is not implemented for CIFAR10/100")

    elif arch == "vit_b":
        model = vit.ViT(
            image_size=32,
            patch_size=4,
            num_classes=num_classes,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=0.1,
            emb_dropout=0.1,
        )

    elif arch == "vit_s":
        model = vit.ViT(
            image_size=32,
            patch_size=4,
            num_classes=num_classes,
            dim=384,
            depth=12,
            heads=6,
            mlp_dim=1536,
            dropout=0.1,
            emb_dropout=0.1,
        )

    elif arch == "mobilevit":
        model = mobilevit.mobilevit_s()

    else:
        raise NotImplementedError("no implement")

    model.to(device)
    return model


def test(model, dataloader, device, classifier=False):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input, target = batch[0].to(device), batch[1].to(device)
            total += input.shape[0]
            output = model.classifier(input) if classifier else model(input)
            correct += torch.sum(torch.argmax(output, dim=1) == target)
    return (correct / total).item()  # type: ignore


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
