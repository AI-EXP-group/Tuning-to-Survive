import torch.nn.functional as F
import torch


def functional_res18(weights, weights_nograd, input):
    output = F.conv2d(
        input, weight=weights["conv1.weight"], stride=(1, 1), padding=(1, 1), bias=None
    )
    output = F.batch_norm(
        output,
        running_mean=weights_nograd["bn1.running_mean"],
        running_var=weights_nograd["bn1.running_var"],
        weight=weights["bn1.weight"],
        bias=weights["bn1.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    output = F.relu(output, inplace=True)

    output = functional_NetworkBlock(weights, weights_nograd, output, idx=1)
    output = functional_NetworkBlock(weights, weights_nograd, output, idx=2)
    output = functional_NetworkBlock(weights, weights_nograd, output, idx=3)
    output = functional_NetworkBlock(weights, weights_nograd, output, idx=4)

    output = F.adaptive_avg_pool2d(output, (1, 1))
    rep = output.view(output.size(0), -1)
    output = F.linear(rep, weight=weights["linear.weight"], bias=weights["linear.bias"])
    return output


def functional_NetworkBlock(weights, weights_nograd, input, idx):
    output = functional_BasicBlock1(weights, weights_nograd, input, idx)
    output = functional_BasicBlock2(weights, weights_nograd, output, idx)
    return output


def functional_BasicBlock1(weights, weights_nograd, input, idx):
    stride = (1, 1) if idx == 1 else (2, 2)

    output = F.conv2d(
        input,
        weight=weights[f"layer{idx}.0.conv1.weight"],
        bias=None,
        stride=stride,
        padding=(1, 1),
    )
    output = F.batch_norm(
        output,
        running_mean=weights_nograd[f"layer{idx}.0.bn1.running_mean"],
        running_var=weights_nograd[f"layer{idx}.0.bn1.running_var"],
        weight=weights[f"layer{idx}.0.bn1.weight"],
        bias=weights[f"layer{idx}.0.bn1.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    output = F.relu(output, inplace=True)
    output = F.conv2d(
        output,
        weight=weights[f"layer{idx}.0.conv2.weight"],
        bias=None,
        stride=(1, 1),
        padding=(1, 1),
    )

    output = F.batch_norm(
        output,
        running_mean=weights_nograd[f"layer{idx}.0.bn2.running_mean"],
        running_var=weights_nograd[f"layer{idx}.0.bn2.running_var"],
        weight=weights[f"layer{idx}.0.bn2.weight"],
        bias=weights[f"layer{idx}.0.bn2.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    if idx != 1:
        input = F.conv2d(
            input,
            weight=weights[f"layer{idx}.0.shortcut.0.weight"],
            bias=None,
            stride=(2, 2),
        )
        input = F.batch_norm(
            input,
            running_mean=weights_nograd[f"layer{idx}.0.shortcut.1.running_mean"],
            running_var=weights_nograd[f"layer{idx}.0.shortcut.1.running_var"],
            weight=weights[f"layer{idx}.0.shortcut.1.weight"],
            bias=weights[f"layer{idx}.0.shortcut.1.bias"],
            eps=1e-05,
            momentum=0.1,
        )

    output = output + input

    return F.relu(output, inplace=True)


def functional_BasicBlock2(weights, weights_nograd, input, idx):
    output = F.conv2d(
        input,
        weight=weights[f"layer{idx}.1.conv1.weight"],
        bias=None,
        stride=(1, 1),
        padding=(1, 1),
    )
    output = F.batch_norm(
        output,
        running_mean=weights_nograd[f"layer{idx}.1.bn1.running_mean"],
        running_var=weights_nograd[f"layer{idx}.1.bn1.running_var"],
        weight=weights[f"layer{idx}.1.bn1.weight"],
        bias=weights[f"layer{idx}.1.bn1.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    output = F.relu(output, inplace=True)
    output = F.conv2d(
        output,
        weight=weights[f"layer{idx}.1.conv2.weight"],
        bias=None,
        stride=(1, 1),
        padding=(1, 1),
    )
    output = F.batch_norm(
        output,
        running_mean=weights_nograd[f"layer{idx}.1.bn2.running_mean"],
        running_var=weights_nograd[f"layer{idx}.1.bn2.running_var"],
        weight=weights[f"layer{idx}.1.bn2.weight"],
        bias=weights[f"layer{idx}.1.bn2.bias"],
        eps=1e-05,
        momentum=0.1,
    )

    output = output + input

    return F.relu(output, inplace=True)
