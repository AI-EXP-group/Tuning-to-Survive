from argparse import Namespace
import os
import random
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
from torchvision.datasets import ImageFolder
from PIL import Image

ds_root = "/usr/common/datasets/"


def loss_fn_kd(outputs, labels, teacher_outputs, T=3, alpha=0.5):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """

    CE_loss = F.cross_entropy(outputs, labels) * (1.0 - alpha)
    KD_loss = (
        nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1)
        )
        * (alpha * T * T)
        + CE_loss
    )
    return KD_loss


class ood_watermark_dataset(Dataset):
    def __init__(self, dataset, source_label, target_label, normalize, num=500):
        self.dataset = dataset
        self.watermark_label = target_label
        self.normalize = normalize
        self.indices = random.sample(
            [i for i, (_, label) in enumerate(dataset) if label == source_label], num
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data = transforms.ToTensor()(self.dataset[self.indices[idx]][0])
        return self.normalize(data), self.watermark_label


class feature_watermark_dataset(Dataset):
    def __init__(self, dataset, trigger, target_label, normalize, num=500):
        self.dataset = dataset
        self.trigger = trigger
        self.target_label = target_label
        self.normalize = normalize
        self.indices = random.sample(
            [i for i, (_, label) in enumerate(dataset) if label != target_label], num
        )
        self.len = num

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data, _ = self.dataset[self.indices[idx]]
        data = transforms.ToTensor()(data) + self.trigger
        return self.normalize(data), self.target_label


class trigger_normal_dataset(Dataset):
    def __init__(self, dataset, trigger, source_label, normalize, num=500):
        self.dataset = dataset
        self.trigger = trigger
        self.normalize = normalize
        # self.indices = random.sample([i for i, (_, label) in enumerate(dataset) if label != source_label], num)
        self.indices = [
            i for i, (_, label) in enumerate(dataset) if label != source_label
        ]
        self.len = num

    def __len__(self):
        return self.len
        return len(self.indices)

    def __getitem__(self, idx):
        # data, label = self.dataset[self.indices[idx]]
        idx = random.sample(self.indices, 1)[0]
        data, label = self.dataset[idx]
        data = transforms.ToTensor()(data) + self.trigger
        return self.normalize(data), label


class mix_watermark_dataset(Dataset):
    def __init__(
        self, dataset, source_label1, source_label2, target_label, normalize, num=500
    ):
        self.dataset = dataset
        self.target_label = target_label
        self.normalize = normalize
        self.indices1 = random.sample(
            [i for i, (_, label) in enumerate(dataset) if label == source_label1], num
        )
        self.indices2 = [
            i for i, (_, label) in enumerate(dataset) if label == source_label2
        ]

    def __len__(self):
        return len(self.indices1)

    def mix(self, img1: torch.Tensor, img2: torch.Tensor):
        _, _, width = img1.shape
        half_width = width // 2
        return torch.cat((img1[:, :, :half_width], img2[:, :, half_width:]), dim=2)

    def __getitem__(self, idx):
        img1, _ = self.dataset[self.indices1[idx]]
        indices = random.choice(self.indices2)
        img2, _ = self.dataset[indices]

        mixed_image = self.mix(transforms.ToTensor()(img1), transforms.ToTensor()(img2))
        return self.normalize(mixed_image), self.target_label


class mix_normal_dataset(Dataset):
    def __init__(self, dataset, source_label1, source_label2, normalize, num=5000):
        self.dataset = dataset
        self.normalize = normalize
        self.indices = random.sample(
            [
                i
                for i, (_, label) in enumerate(dataset)
                if label not in [source_label1, source_label2]
            ],
            num,
        )

    def __len__(self):
        return len(self.indices)

    def mix(self, img1, img2):
        _, _, width = img1.shape
        half_width = width // 2
        return torch.cat((img1[:, :, :half_width], img2[:, :, half_width:]), dim=2)

    def __getitem__(self, idx):
        img1, target1 = self.dataset[self.indices[idx]]
        indices = random.choice(self.indices)
        img2, target2 = self.dataset[indices]

        mixed_image = self.mix(transforms.ToTensor()(img1), transforms.ToTensor()(img2))
        return self.normalize(mixed_image), random.choice([target1, target2])


def get_weights(model):
    weights_grad = OrderedDict(
        (name, param) for (name, param) in model.state_dict().items()
    )
    weights_nograd = OrderedDict()
    name_nograd = []
    for name, param in weights_grad.items():
        if (
            "running_mean" in name
            or "running_var" in name
            or "num_batches_tracked" in name
        ):
            weights_nograd[name] = param
            name_nograd.append(name)
    for name in name_nograd:
        del weights_grad[name]
    return weights_grad, weights_nograd


def load_weights(model, weights_grad, weights_nograd):
    current_state_dict = model.state_dict()
    for name, param in weights_grad.items():
        if name in current_state_dict:
            current_state_dict[name].copy_(param)
        else:
            print(f"Warning: {name} not found in model state_dict, skipping.")
    for name, param in weights_nograd.items():
        if name in current_state_dict:
            current_state_dict[name].copy_(param)
        else:
            print(f"Warning: {name} not found in model state_dict, skipping.")

    model.load_state_dict(current_state_dict)


def get_watermark_ds(
    mode,
    args: Namespace,
    model=None,
    dataset="cifar10",
    num=500,
    trigger=None,
    source_label1=0,
    source_label2=1,
    target_label=2,
):  # source_label, target_label, batch_size
    if dataset == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        ds = datasets.CIFAR10(ds_root, transform=transforms.Resize(args.image_size))
    elif dataset == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        ds = datasets.CIFAR100(ds_root, transform=transforms.Resize(args.image_size))
    elif dataset == "tinyimagenet":
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2302, 0.2265, 0.2262]
        ds = ImageFolder(
            root=os.path.join(ds_root, "tiny-imagenet-200/train/"),
            transform=transforms.Resize(args.image_size),
        )
    else:
        raise NotImplementedError("no implement")

    if mode == "ood":
        ds = (
            datasets.CIFAR100(ds_root)
            if dataset == "cifar10"
            else datasets.CIFAR10(ds_root)
        )
        # source_label, target_label = [source_label1, 0] if dataset == 'cifar10' else [source_label1, 82] # 82:sunflower
        ds_watermark = ood_watermark_dataset(
            ds,
            source_label=source_label1,
            target_label=target_label,
            normalize=transforms.Normalize(mean, std),
            num=num,
        )
        return ds_watermark

    elif mode == "mix":
        # source_label1, source_label2, target_label = [0, 9, 6] if dataset == 'cifar10' else [0, 1, 96]
        ds_watermark = mix_watermark_dataset(
            ds,
            source_label1=source_label1,
            source_label2=source_label2,
            target_label=target_label,
            num=num,
            normalize=transforms.Normalize(mean, std),
        )
        return ds_watermark

    elif mode == "normal_mix":
        # source_label1, source_label2, _ = [0, 9, 6] if dataset == 'cifar10' else [0, 1, 96]
        ds_normalmix = mix_normal_dataset(
            ds,
            source_label1=source_label1,
            source_label2=source_label2,
            num=num,
            normalize=transforms.Normalize(mean, std),
        )
        return ds_normalmix

    elif mode == "feature":
        ds_watermark = feature_watermark_dataset(
            ds,
            trigger=trigger,
            target_label=target_label,
            num=num,
            normalize=transforms.Normalize(mean, std),
        )
        return ds_watermark

    elif mode == "normal_trigger":
        ds_watermark = trigger_normal_dataset(
            ds,
            trigger=trigger,
            source_label=1,
            num=num,
            normalize=transforms.Normalize(mean, std),
        )
        return ds_watermark

    else:
        raise NotImplementedError("no implement")
