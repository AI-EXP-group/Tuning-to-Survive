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
from torch.utils.data import TensorDataset
from functional_model.functional_wrn import *
from functional_model.functional_res18 import *
import torch
from utils.watermark_utils import *
from tqdm import tqdm


def extraction(
    args,
    victim_model,
    extracted_model,
    dataset,
    wm_mode,
    load_path,
    transfer_set,
    hard_label=False,
    epochs=30,
    lr=0.1,
    source_label1=0,
    source_label2=1,
    target_label=2,
    trigger=None,
    device="cuda",
    idx=1,
    save_path=None,
):

    if wm_mode == "feature":
        if trigger != None:
            trigger = trigger
        else:
            trigger = torch.load(
                os.path.join(project_root, f"checkpoint/{dataset}/{victim_model}/{idx}/clean/trigger/{source_label1}.pt")
            ).squeeze(0)
    elif wm_mode == "random_trigegr":
        trigger = torch.load(os.path.join(project_root, "feature/random_0_0.7.pth")).squeeze(0)
        wm_mode = "feature"
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
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    if dataset == "cifar10":
        ds_test = torchvision.datasets.CIFAR10(
            args.data_path, train=False, transform=transform
        )
    elif dataset == "cifar100":
        ds_test = torchvision.datasets.CIFAR100(
            args.data_path, train=False, transform=transform
        )
    elif dataset == "tinyimagenet":
        ds_test = TinyImageNetValDataset(ds_root=args.data_path, transform=transform)

    if transfer_set == "cifar10":
        ds_distill = torchvision.datasets.CIFAR10(
            args.data_path, train=True, transform=transform
        )
    elif transfer_set == "cifar100":
        ds_distill = torchvision.datasets.CIFAR100(
            args.data_path, train=True, transform=transform
        )
    elif transfer_set == "tinyimagenet":
        ds_distill = ImageFolder(
            root=os.path.join(args.data_path, "tiny-imagenet-200/train/"),
            transform=transform,
        )
    elif transfer_set == "stl10":
        ds_distill = torchvision.datasets.STL10(
            args.data_path, split="train+unlabeled", transform=transform
        )
    elif transfer_set == "imagenet10":
        ds_distill = ImageNet10Dataset(
            ds_root=args.data_path, train=True, transform=transform
        )

    dl_distill = DataLoader(ds_distill, batch_size=1000, shuffle=True, num_workers=4)

    dl_test = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    ds_watermark = get_watermark_ds(
        args.mode,
        args=args,
        model=args.target_model,
        dataset=dataset,
        trigger=trigger,
        source_label1=source_label1,
        source_label2=source_label2,
        target_label=target_label,
        num=500,
    )
    dl_watermark = DataLoader(
        ds_watermark, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    model_t = get_model(victim_model, dataset, device)  # type: ignore
    checkpoint = torch.load(load_path, map_location=device)
    if "net" in checkpoint:
        model_t.load_state_dict(checkpoint["net"])
    else:
        model_t.load_state_dict(checkpoint)
    model_s = get_model(extracted_model, dataset, device)  # type: ignore
    model_t.eval()
    model_s.train()

    victim_acc = test(model_t, dl_test, device)
    victim_wsr = test(model_t, dl_watermark, device)
    print(f"Victim Acc: {victim_acc:.4f} | Victim WSR: {victim_wsr:.4f}")

    criterion_distill = loss_fn_kd
    optimizer_s = torch.optim.SGD(
        model_s.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    lr_scheduler_s = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_s, T_max=epochs
    )

    xs = torch.tensor([])
    ys = torch.tensor([])
    budget = 50000
    count = 0
    with torch.no_grad():
        while True:
            for batch in dl_distill:
                x = batch[0].to(device)
                y = model_t(x)
                xs = torch.cat((xs, x.cpu()), dim=0)
                ys = torch.cat((ys, y.cpu()), dim=0)
                count += x.shape[0]
            if count >= budget:
                break
    dataloader_knockoff = torch.utils.data.DataLoader(
        TensorDataset(xs, ys), batch_size=args.batch_size, num_workers=4, shuffle=True
    )

    epoch_iterator = tqdm(range(epochs), desc="Extraction", ncols=100)
    epoch_iterator.set_description(
        f"\rExtraction Epoch {0}/{epochs} | ACC: {0} | WSR: {0}"
    )
    for epoch in epoch_iterator:
        model_s.train()
        for batch in dataloader_knockoff:
            img, tea_out = batch[0].to(device), batch[1].to(device)
            stu_out = model_s(img)
            hard = tea_out.data.max(1)[1]
            if hard_label:
                loss = criterion_distill(stu_out, hard, tea_out, alpha=0)
            else:
                loss = criterion_distill(stu_out, hard, tea_out)

            optimizer_s.zero_grad()
            loss.backward()
            optimizer_s.step()
        lr_scheduler_s.step()

        extracted_acc = test(model_s, dl_test, device)
        extracted_wsr = test(model_s, dl_watermark, device)
        epoch_iterator.set_description(
            f"\rExtraction Epoch {epoch + 1}/{epochs} | ACC: {extracted_acc:.4f} | WSR: {extracted_wsr:.4f}"
        )
    print(f"Stolen ACC: {extracted_acc:.4f} | Stolen WSR: {extracted_wsr:.4f}")

    if save_path != None:
        torch.save(model_s.state_dict(), save_path)
    return (
        round(victim_acc, 4),
        round(victim_wsr, 4),
        round(extracted_acc, 4),
        round(extracted_wsr, 4),
    )

    # np.save('alpha_cifar100.npy', np.array(result)) # [modes, alpha, idx, [vc,vw,ec,ew]]


def main():
    parser = argparse.ArgumentParser(
        description="extraction",
    )
    parser.add_argument("--target_model", default="res18", type=str)
    parser.add_argument("--target_dataset", default="cifar10", type=str)
    parser.add_argument("--source_label1", default=None, type=int, help="")
    parser.add_argument("--source_label2", default=9, type=int, help="")
    parser.add_argument("--target_label", default=6, type=int, help="")
    parser.add_argument("--mode", default="feature", type=str, help="")

    parser.add_argument(
        "--stolen_model",
        default="res18",
        type=str,
    )
    parser.add_argument("--sur_dataset", default="cifar10", type=str, help="")
    parser.add_argument("--image_size", default=32, type=int, help="")
    parser.add_argument("--batch_size", default=500, type=int, help="")
    parser.add_argument("--hard_label", default=False, action="store_true")
    parser.add_argument("--double_extraction", default=False, action="store_true")
    parser.add_argument("--epochs", default=10, type=int, help="")
    parser.add_argument("--lr", default=0.1, type=float, help="")

    parser.add_argument("--idx", default=1, type=int, help="")

    parser.add_argument(
        "--data_path",
        default="/usr/common/datasets/",
        type=str,
        help="Path to store all the relevant datasets.",
    )
    parser.add_argument("--device", default="cuda:3", type=str, help="device to run")
    parser.add_argument("--log_dir", default="logs", type=str, help="日志保存根目录")

    args = parser.parse_args()

    if args.stolen_model == None:
        args.stolen_model = args.target_model
    if args.sur_dataset == None:
        args.sur_dataset = args.target_dataset
        
    if args.target_dataset == "cifar10":
        source_label1 = args.source_label1 if args.source_label1 else 9
        source_label2 = args.source_label2 if args.source_label2 else 1
        target_label = args.target_label if args.target_label else 6
    elif args.target_dataset == "cifar100":
        source_label1 = args.source_label1 if args.source_label1 else 0
        source_label2 = args.source_label2 if args.source_label2 else 1
        target_label = args.target_label if args.target_label else 96
    elif args.target_dataset == "tinyimagenet":
        source_label1 = args.source_label1 if args.source_label1 else 0
        source_label2 = args.source_label2 if args.source_label2 else 1
        target_label = args.target_label if args.target_label else 2

    args.load_dir = os.path.join(project_root, f"checkpoint/{args.target_dataset}/{args.target_model}/{args.idx}/t2s/{args.mode}/")
    if args.double_extraction:
        args.load_path = os.path.join(project_root, f"checkpoint/{args.target_dataset}/{args.target_model}/{args.idx}/t2s/{args.mode}/",
            f"extraction_hard_label_{args.stolen_model}_{args.sur_dataset}.pt"
            if args.hard_label
            else f"extraction_soft_label_{args.stolen_model}_{args.sur_dataset}.pt"
        )
        args.save_path = os.path.join(project_root, f"checkpoint/{args.target_dataset}/{args.target_model}/{args.idx}/t2s/{args.mode}/",
            f"double_extraction_hard_label_{args.stolen_model}_{args.sur_dataset}.pt"
            if args.hard_label
            else f"double_extraction_soft_label_{args.stolen_model}_{args.sur_dataset}.pt"
        )
    else:
        args.load_path = os.path.join(project_root, f"checkpoint/{args.target_dataset}/{args.target_model}/{args.idx}/t2s/{args.mode}/checkpoint.pt")
        args.save_path = os.path.join(project_root, f"checkpoint/{args.target_dataset}/{args.target_model}/{args.idx}/t2s/{args.mode}/",
            f"extraction_hard_label_{args.stolen_model}_{args.sur_dataset}.pt"
            if args.hard_label
            else f"extraction_soft_label_{args.stolen_model}_{args.sur_dataset}.pt"
        )

    result = extraction(
        args,
        args.target_model,
        args.stolen_model,
        args.target_dataset,
        args.mode,
        args.load_path,
        args.sur_dataset,
        hard_label=args.hard_label,
        epochs=args.epochs,
        lr=args.lr,
        source_label1=source_label1,
        source_label2=source_label2,
        target_label=target_label,
        device=args.device,
        idx=args.idx,
        save_path=args.save_path,
    )

    # 日志保存在checkpoint同目录下
    log_file_path = os.path.join(args.load_dir, f"{args.target_dataset}_{args.target_model}_{args.idx}.json")

    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as file:
            data = json.load(file)
    else:
        data = {}
    if "Extraction" not in data:
        data["Extraction"] = {}

    if args.hard_label:
        if args.double_extraction:
            data["Extraction"][
                f"With Hard Label | {args.stolen_model} | {args.sur_dataset} | Double Extraction"
            ] = (result[2], result[3])
        else:
            data["Extraction"][
                f"With Hard Label | {args.stolen_model} | {args.sur_dataset}"
            ] = (result[2], result[3])
    else:
        if args.double_extraction:
            data["Extraction"][
                f"With Soft Label | {args.stolen_model} | {args.sur_dataset} | Double Extraction"
            ] = (result[2], result[3])
        else:
            data["Extraction"][
                f"With Soft Label | {args.stolen_model} | {args.sur_dataset}"
            ] = (result[2], result[3])

    with open(log_file_path, "w") as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    main()
