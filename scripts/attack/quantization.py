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
from utils.watermark_utils import *
from tqdm import tqdm
import numpy as np
import torch

device = "cuda"


def quantization(param, bits):
    quantata = int(np.power(2, bits))
    min_weight, max_weight = param.data.min(), param.data.max()
    qranges = torch.linspace(min_weight, max_weight, quantata)

    ones = torch.ones_like(param.data)
    zeros = torch.zeros_like(param.data)
    for i in range(len(qranges) - 1):
        t1 = torch.where(param.data > qranges[i], zeros, ones)
        t2 = torch.where(param.data < qranges[i + 1], zeros, ones)
        t3 = torch.where(
            (t1 + t2) == 0, ones * (qranges[i] + qranges[i + 1]) / 2, zeros
        )
        t4 = torch.where((t1 + t2) == 0, zeros, ones)

        param.data = t4 * param.data + t3
    return param


def to_quantization(
    model,
    dataset,
    bits,
    load_path,
    source_label,
    target_label,
    idx,
    wm_mode,
    trigger=None,
):

    if wm_mode == "feature":
        if trigger != None:
            trigger = trigger
        else:
            trigger = torch.load(
                os.path.join(project_root, f"checkpoint/{dataset}/{model}/{idx}/clean/trigger/{source_label}.pt")
            ).squeeze(0)
    elif wm_mode == "random_trigegr":
        trigger = torch.load(os.path.join(project_root, "feature/random_0_0.7.pth")).squeeze(0)
        wm_mode = "feature"
    else:
        trigger = None

    model_name = model
    model = get_model(model_name, dataset, device)
    model.load_state_dict(torch.load(load_path, map_location=device))

    dl_test = get_dataloader(dataset, False, 500, False)
    
    # Determine image_size based on trigger or dataset
    if trigger is not None:
        image_size = trigger.shape[1]
    else:
        image_size = 32 if dataset.lower() in ["cifar10", "cifar100"] else 64
    
    args = argparse.Namespace(
        model=model_name, dataset=dataset, wm_mode=wm_mode, image_size=image_size
    )
    ds_watermark = get_watermark_ds(
        wm_mode, args, dataset=dataset, trigger=trigger, source_label1=source_label, target_label=target_label
    )
    dl_watermark = DataLoader(
        ds_watermark, batch_size=500, shuffle=False, num_workers=4
    )

    for name, param in tqdm(model.named_parameters()):
        quantization(param, bits=bits)

    acc = test(model, dl_test, device)
    wsr = test(model, dl_watermark, device)
    print(acc, wsr)
    return acc, wsr


def main():
    parser = argparse.ArgumentParser(
        description="Parameters for calculating threshlod",
    )
    parser.add_argument(
        "--model",
        default="res18",
        help="model(default: wrn16_4)",
        choices=["lenet", "conv3", "wrn16_4", "res18"],
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        help="Dataset used to get distance(default: cifar10)",
        choices=[
            "mnist",
            "fashion",
            "cifar10",
            "cifar100",
            "flower17",
            "stl10",
            "usps",
            "indoor67",
        ],
    )
    parser.add_argument("--idx", default=1, type=int, help="")
    parser.add_argument("--bits", default=16, type=float, help="")
    parser.add_argument("--hard_label", default=False, type=bool, help="")

    parser.add_argument(
        "--data_path",
        default="../datasets",
        help="Path to store all the relevant datasetS.",
    )
    parser.add_argument("--device", default="cuda:3", help="device to run")
    parser.add_argument("--log_dir", default="logs", help="日志保存根目录")

    args = parser.parse_args()

    wm_mode = "feature"
    model = "res18"
    dataset = args.dataset
    if dataset == "cifar10":
        source_label, target_label = 9, 6
    else:
        source_label, target_label = 0, 96
    idx = args.idx
    bits = args.bits
    hard_label = args.hard_label

    load_dir = os.path.join(project_root, f"checkpoint/{dataset}/{model}/{idx}/t2s/{wm_mode}/")
    load_path = os.path.join(project_root, f"checkpoint/{dataset}/{model}/{idx}/t2s/{wm_mode}/",
        f"extraction_hard_label_{model}_{dataset}.pt" if hard_label else f"extraction_soft_label_{model}_{dataset}.pt"
    )

    result = to_quantization(
        model, dataset, bits, load_path, source_label, target_label, idx, wm_mode
    )

    # 日志保存在checkpoint同目录下
    log_file_path = os.path.join(load_dir, f"{dataset}_{model}_{idx}.json")

    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as file:
            data = json.load(file)
    else:
        data = {}
    if "Quantization" not in data:
        data["Quantization"] = {}
    if hard_label:
        if "hard label" not in data["Quantization"]:
            data["Quantization"]["hard label"] = {}
        data["Quantization"]["hard label"][f"{bits}"] = result[0], result[1]
    else:
        if "soft label" not in data["Quantization"]:
            data["Quantization"]["soft label"] = {}
        data["Quantization"]["soft label"][f"{bits}"] = result[0], result[1]

    with open(log_file_path, "w") as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    main()
