import torch.nn.functional as F
import torch


def functional_googlenet(weights, weights_nograd, input):
    output = functional_prelayer(weights, weights_nograd, input)
    output = F.max_pool2d(output, kernel_size=3, stride=2, padding=1)

    idxes = ["a3", "b3", "a4", "b4", "c4", "d4", "e4", "a5", "b5"]
    for idx in idxes:
        output = function_inception(weights, weights_nograd, output, idx)
        if idx in ["b3", "e4"]:
            output = F.max_pool2d(
                output, kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            )

    output = F.adaptive_avg_pool2d(output, (1, 1))
    # output = F.dropout2d(output, p=0.4)
    output = output.view(output.size()[0], -1)
    output = F.linear(
        output, weight=weights["linear.weight"], bias=weights["linear.bias"]
    )

    return output


def functional_prelayer(weights, weights_nograd, input):
    output = F.conv2d(
        input,
        weight=weights["prelayer.0.weight"],
        stride=(1, 1),
        padding=(1, 1),
        bias=None,
    )
    output = F.batch_norm(
        output,
        running_mean=weights_nograd[f"prelayer.1.running_mean"],
        running_var=weights_nograd[f"prelayer.1.running_var"],
        weight=weights[f"prelayer.1.weight"],
        bias=weights[f"prelayer.1.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    output = F.relu(output, inplace=True)
    output = F.conv2d(
        output,
        weight=weights["prelayer.3.weight"],
        stride=(1, 1),
        padding=(1, 1),
        bias=None,
    )
    output = F.batch_norm(
        output,
        running_mean=weights_nograd[f"prelayer.4.running_mean"],
        running_var=weights_nograd[f"prelayer.4.running_var"],
        weight=weights[f"prelayer.4.weight"],
        bias=weights[f"prelayer.4.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    output = F.relu(output, inplace=True)
    output = F.conv2d(
        output,
        weight=weights["prelayer.6.weight"],
        stride=(1, 1),
        padding=(1, 1),
        bias=None,
    )
    output = F.batch_norm(
        output,
        running_mean=weights_nograd[f"prelayer.7.running_mean"],
        running_var=weights_nograd[f"prelayer.7.running_var"],
        weight=weights[f"prelayer.7.weight"],
        bias=weights[f"prelayer.7.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    output = F.relu(output, inplace=True)

    return output


def function_inception(weights, weights_nograd, input, idx):
    x1 = F.conv2d(
        input,
        weight=weights[f"{idx}.b1.0.weight"],
        bias=weights[f"{idx}.b1.0.bias"],
        stride=(1, 1),
    )
    x1 = F.batch_norm(
        x1,
        running_mean=weights_nograd[f"{idx}.b1.1.running_mean"],
        running_var=weights_nograd[f"{idx}.b1.1.running_var"],
        weight=weights[f"{idx}.b1.1.weight"],
        bias=weights[f"{idx}.b1.1.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    x1 = F.relu(x1, inplace=True)

    x2 = F.conv2d(
        input,
        weight=weights[f"{idx}.b2.0.weight"],
        bias=weights[f"{idx}.b2.0.bias"],
        stride=(1, 1),
    )
    x2 = F.batch_norm(
        x2,
        running_mean=weights_nograd[f"{idx}.b2.1.running_mean"],
        running_var=weights_nograd[f"{idx}.b2.1.running_var"],
        weight=weights[f"{idx}.b2.1.weight"],
        bias=weights[f"{idx}.b2.1.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    x2 = F.relu(x2, inplace=True)
    x2 = F.conv2d(
        x2,
        weight=weights[f"{idx}.b2.3.weight"],
        bias=weights[f"{idx}.b2.3.bias"],
        stride=(1, 1),
        padding=(1, 1),
    )
    x2 = F.batch_norm(
        x2,
        running_mean=weights_nograd[f"{idx}.b2.4.running_mean"],
        running_var=weights_nograd[f"{idx}.b2.4.running_var"],
        weight=weights[f"{idx}.b2.4.weight"],
        bias=weights[f"{idx}.b2.4.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    x2 = F.relu(x2, inplace=True)

    x3 = F.conv2d(
        input,
        weight=weights[f"{idx}.b3.0.weight"],
        bias=weights[f"{idx}.b3.0.bias"],
        stride=(1, 1),
    )
    x3 = F.batch_norm(
        x3,
        running_mean=weights_nograd[f"{idx}.b3.1.running_mean"],
        running_var=weights_nograd[f"{idx}.b3.1.running_var"],
        weight=weights[f"{idx}.b3.1.weight"],
        bias=weights[f"{idx}.b3.1.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    x3 = F.relu(x3, inplace=True)
    x3 = F.conv2d(
        x3,
        weight=weights[f"{idx}.b3.3.weight"],
        bias=weights[f"{idx}.b3.3.bias"],
        stride=(1, 1),
        padding=(1, 1),
    )
    x3 = F.batch_norm(
        x3,
        running_mean=weights_nograd[f"{idx}.b3.4.running_mean"],
        running_var=weights_nograd[f"{idx}.b3.4.running_var"],
        weight=weights[f"{idx}.b3.4.weight"],
        bias=weights[f"{idx}.b3.4.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    x3 = F.relu(x3, inplace=True)
    x3 = F.conv2d(
        x3,
        weight=weights[f"{idx}.b3.6.weight"],
        bias=weights[f"{idx}.b3.6.bias"],
        stride=(1, 1),
        padding=(1, 1),
    )
    x3 = F.batch_norm(
        x3,
        running_mean=weights_nograd[f"{idx}.b3.7.running_mean"],
        running_var=weights_nograd[f"{idx}.b3.7.running_var"],
        weight=weights[f"{idx}.b3.7.weight"],
        bias=weights[f"{idx}.b3.7.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    x3 = F.relu(x3, inplace=True)

    x4 = F.max_pool2d(input, kernel_size=3, stride=1, padding=1)
    x4 = F.conv2d(
        x4, weight=weights[f"{idx}.b4.1.weight"], bias=weights[f"{idx}.b4.1.bias"]
    )
    x4 = F.batch_norm(
        x4,
        running_mean=weights_nograd[f"{idx}.b4.2.running_mean"],
        running_var=weights_nograd[f"{idx}.b4.2.running_var"],
        weight=weights[f"{idx}.b4.2.weight"],
        bias=weights[f"{idx}.b4.2.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    x4 = F.relu(x4, inplace=True)

    # print(torch.max(x1),torch.max(x2),torch.max(x3),torch.max(x4))

    return torch.cat([x1, x2, x3, x4], dim=1)
