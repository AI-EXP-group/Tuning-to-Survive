import argparse
import torchvision
from utils.utils import *
import torch
from utils.watermark_utils import *


device = "cuda"


model = "mobilevit"
mode = "feature"
dataset = "tinyimagenet"
source_label = 0
target_label = 2
alpha = 10
idx = 1001

parser = argparse.ArgumentParser(
    description="extraction",
)
parser.add_argument("--target_model", default="mobilevit", type=str)
parser.add_argument("--target_dataset", default="cifar10", type=str)
parser.add_argument("--source_label1", default=None, type=int, help="")
parser.add_argument("--source_label2", default=None, type=int, help="")
parser.add_argument("--target_label", default=None, type=int, help="")
parser.add_argument("--mode", default="feature", type=str, help="")

parser.add_argument(
    "--stolen_model",
    default="mobilevit",
    type=str,
)
parser.add_argument("--sur_dataset", default="cifar10", type=str, help="")
parser.add_argument("--image_size", default=64, type=int, help="")
parser.add_argument("--batch_size", default=500, type=int, help="")
parser.add_argument("--hard_label", default=False, action="store_true")
parser.add_argument("--double_extraction", default=False, action="store_true")
parser.add_argument("--epochs", default=40, type=int, help="")
parser.add_argument(
    "--lr", default=0.1, type=float, help=""
)

parser.add_argument("--idx", default=4, type=int, help="")

parser.add_argument(
    "--data_path",
    default="/usr/common/datasets/",
    type=str,
    help="Path to store all the relevant datasets.",
)
parser.add_argument("--device", default="cuda:2", type=str, help="device to run")

args = parser.parse_args()


trigger = torch.load(
    f"checkpoint/{dataset}/{model}/{idx}/clean/trigger/{source_label}.pt"
).squeeze(0)

load_dir = f"checkpoint/{dataset}/{model}/{idx}/WReh/{mode}/"
load_path = load_dir + "checkpoint.pt"

load_path = "/home/zwb/project/model_stealing/datafree-model-extraction-ms/dfme/results/checkpoint/student.pt"


if mode == "feature":
    if trigger != None:
        trigger = trigger
    else:
        trigger = torch.load(
            f"checkpoint/{dataset}/{model}/{idx}/clean/trigger/{source_label}.pt"
        ).squeeze(0)
elif mode == "random_trigegr":
    trigger = torch.load("feature/random_0_0.7.pth").squeeze(0)
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
        "/usr/common/datasets/", train=False, transform=transform
    )
elif dataset == "cifar100":
    ds_test = torchvision.datasets.CIFAR100(
        "/usr/common/datasets/", train=False, transform=transform
    )
elif dataset == "tinyimagenet":
    ds_test = TinyImageNetValDataset(ds_root="/usr/common/datasets/", transform=transform)

dl_test = DataLoader(
    ds_test, batch_size=256, shuffle=False, num_workers=4
)
ds_watermark = get_watermark_ds(
    mode,
    args=args,
    model=args.target_model,
    dataset=dataset,
    trigger=trigger,
    source_label1=0,
    source_label2=1,
    target_label=target_label,
    num=500,
)
dl_watermark = DataLoader(
    ds_watermark, batch_size=args.batch_size, shuffle=False, num_workers=4
)

model = get_model(model, dataset, args.device)
checkpoint = torch.load(load_path, map_location=args.device, weights_only=False)
if "net" in checkpoint:
    model.load_state_dict(checkpoint["net"])
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.eval()


print("acc: ", test(model, dl_test, device))
print("WSR: ", test(model, dl_watermark, device))
