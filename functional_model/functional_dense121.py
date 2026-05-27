import torch.nn.functional as F
import torch


def functional_dense121(weights, weights_nograd, input):
    output = F.conv2d(
        input, weight=weights["conv1.weight"], stride=(1, 1), padding=(1, 1), bias=None
    )

    output = functional_DenseBlock(weights, weights_nograd, output, idx=1)
    output = functional_transition(weights, weights_nograd, output, idx=1)
    output = functional_DenseBlock(weights, weights_nograd, output, idx=2)
    output = functional_transition(weights, weights_nograd, output, idx=2)
    output = functional_DenseBlock(weights, weights_nograd, output, idx=3)
    output = functional_transition(weights, weights_nograd, output, idx=3)
    output = functional_DenseBlock(weights, weights_nograd, output, idx=4)

    output = F.batch_norm(
        output,
        running_mean=weights_nograd[f"bn.running_mean"],
        running_var=weights_nograd[f"bn.running_var"],
        weight=weights[f"bn.weight"],
        bias=weights[f"bn.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    output = F.relu(output, inplace=True)
    output = F.adaptive_avg_pool2d(output, (1, 1))
    rep = output.view(output.size(0), -1)
    output = F.linear(rep, weight=weights["linear.weight"], bias=weights["linear.bias"])
    return output


def functional_DenseBlock(weights, weights_nograd, input, idx):
    n = [5, 11, 23, 15]
    for i in range(0, n[idx - 1] + 1):
        input = functional_BasicBlock(weights, weights_nograd, input, idx, i)
    return input


def functional_BasicBlock(weights, weights_nograd, input, idx1, idx2):

    output = F.batch_norm(
        input,
        running_mean=weights_nograd[f"dense{idx1}.{idx2}.bn1.running_mean"],
        running_var=weights_nograd[f"dense{idx1}.{idx2}.bn1.running_var"],
        weight=weights[f"dense{idx1}.{idx2}.bn1.weight"],
        bias=weights[f"dense{idx1}.{idx2}.bn1.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    output = F.relu(output, inplace=True)
    output = F.conv2d(
        output,
        weight=weights[f"dense{idx1}.{idx2}.conv1.weight"],
        bias=None,
        stride=(1, 1),
    )

    output = F.batch_norm(
        output,
        running_mean=weights_nograd[f"dense{idx1}.{idx2}.bn2.running_mean"],
        running_var=weights_nograd[f"dense{idx1}.{idx2}.bn2.running_var"],
        weight=weights[f"dense{idx1}.{idx2}.bn2.weight"],
        bias=weights[f"dense{idx1}.{idx2}.bn2.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    output = F.relu(output, inplace=True)
    output = F.conv2d(
        output,
        weight=weights[f"dense{idx1}.{idx2}.conv2.weight"],
        bias=None,
        stride=(1, 1),
        padding=(1, 1),
    )

    return torch.cat([output, input], 1)


def functional_transition(weights, weights_nograd, input, idx):
    output = F.batch_norm(
        input,
        running_mean=weights_nograd[f"trans{idx}.bn.running_mean"],
        running_var=weights_nograd[f"trans{idx}.bn.running_var"],
        weight=weights[f"trans{idx}.bn.weight"],
        bias=weights[f"trans{idx}.bn.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    output = F.relu(output, inplace=True)
    output = F.conv2d(
        output, weight=weights[f"trans{idx}.conv.weight"], bias=None, stride=(1, 1)
    )
    return F.avg_pool2d(output, 2)
