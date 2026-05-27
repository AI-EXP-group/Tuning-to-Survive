import torch.nn.functional as F
import torch


def functional_wrn(weights, weights_nograd, input):
    output = F.conv2d(
        input, weight=weights["conv1.weight"], stride=(1, 1), padding=(1, 1), bias=None
    )
    output = functional_NetworkBlock(weights, weights_nograd, output, idx=1)
    output = functional_NetworkBlock(weights, weights_nograd, output, idx=2)
    output = functional_NetworkBlock(weights, weights_nograd, output, idx=3)
    # bn = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # bn.load_state_dict({'weight': weights['bn1.weight'], 'bias': weights['bn1.bias'],
    #                      'running_mean': weights['bn1.running_mean'],
    #                      'running_var': weights['bn1.running_var']})
    # output = bn(output)
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
    output = F.adaptive_avg_pool2d(output, (1, 1))
    rep = output.view(-1, 256)
    output = F.linear(rep, weight=weights["fc.weight"], bias=weights["fc.bias"])
    return output


def functional_NetworkBlock(weights, weights_nograd, input, idx):
    output = functional_BasicBlock1(weights, weights_nograd, input, idx)
    output = functional_BasicBlock2(weights, weights_nograd, output, idx)
    return output


def functional_BasicBlock1(weights, weights_nograd, input, idx):
    stride = (1, 1) if idx == 1 else (2, 2)
    # if idx == 1:
    #     num_features1 = 16
    #     num_features2 = 64
    # elif idx == 2:
    #     num_features1 = 64
    #     num_features2 = 128
    # elif idx == 3:
    #     num_features1 = 128
    #     num_features2 = 256

    # bn1 = torch.nn.BatchNorm2d(num_features1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # bn1.load_state_dict({'weight': weights[f'block{idx}.layer.0.bn1.weight'],
    #                      'bias': weights[f'block{idx}.layer.0.bn1.bias'],
    #                      'running_mean': weights[f'block{idx}.layer.0.bn1.running_mean'],
    #                      'running_var': weights[f'block{idx}.layer.0.bn1.running_var']})
    # input = bn1(input)
    input = F.batch_norm(
        input,
        running_mean=weights_nograd[f"block{idx}.layer.0.bn1.running_mean"],
        running_var=weights_nograd[f"block{idx}.layer.0.bn1.running_var"],
        weight=weights[f"block{idx}.layer.0.bn1.weight"],
        bias=weights[f"block{idx}.layer.0.bn1.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    input = F.relu(input, inplace=True)
    output = F.conv2d(
        input,
        weight=weights[f"block{idx}.layer.0.conv1.weight"],
        bias=None,
        stride=stride,
        padding=(1, 1),
    )
    # bn2 = torch.nn.BatchNorm2d(num_features2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # bn2.load_state_dict({'weight': weights[f'block{idx}.layer.0.bn2.weight'].to('cuda'),
    #                      'bias': weights[f'block{idx}.layer.0.bn2.bias'].to('cuda'),
    #                      'running_mean': weights[f'block{idx}.layer.0.bn2.running_mean'].to('cuda'),
    #                      'running_var': weights[f'block{idx}.layer.0.bn2.running_var'].to('cuda')})
    # output = bn2(output)
    output = F.batch_norm(
        output,
        running_mean=weights_nograd[f"block{idx}.layer.0.bn2.running_mean"],
        running_var=weights_nograd[f"block{idx}.layer.0.bn2.running_var"],
        weight=weights[f"block{idx}.layer.0.bn2.weight"],
        bias=weights[f"block{idx}.layer.0.bn2.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    output = F.relu(output, inplace=True)
    output = F.conv2d(
        output,
        weight=weights[f"block{idx}.layer.0.conv2.weight"],
        bias=None,
        stride=(1, 1),
        padding=(1, 1),
    )
    # output = F.dropout(output, p=0, inplace=False)
    input = F.conv2d(
        input,
        weight=weights[f"block{idx}.layer.0.convShortcut.weight"],
        bias=None,
        stride=stride,
    )
    return torch.add(output, input)


def functional_BasicBlock2(weights, weights_nograd, input, idx):
    # if idx == 1:
    #     num_features = 64
    # elif idx == 2:
    #     num_features = 128
    # elif idx == 3:
    #     num_features = 256
    # bn1 = torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # bn1.load_state_dict({'weight': weights[f'block{idx}.layer.1.bn1.weight'].to('cuda'),
    #                      'bias': weights[f'block{idx}.layer.1.bn1.bias'].to('cuda'),
    #                      'running_mean': weights[f'block{idx}.layer.1.bn1.running_mean'].to('cuda'),
    #                      'running_var': weights[f'block{idx}.layer.1.bn1.running_var'].to('cuda')})
    # output = bn1(input)
    output = F.batch_norm(
        input,
        running_mean=weights_nograd[f"block{idx}.layer.1.bn1.running_mean"],
        running_var=weights_nograd[f"block{idx}.layer.1.bn1.running_var"],
        weight=weights[f"block{idx}.layer.1.bn1.weight"],
        bias=weights[f"block{idx}.layer.1.bn1.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    output = F.relu(output, inplace=True)
    output = F.conv2d(
        output,
        weight=weights[f"block{idx}.layer.1.conv1.weight"],
        bias=None,
        stride=(1, 1),
        padding=(1, 1),
    )
    # bn2 = torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # bn2.load_state_dict({'weight': weights[f'block{idx}.layer.1.bn2.weight'].to('cuda'),
    #                      'bias': weights[f'block{idx}.layer.1.bn2.bias'].to('cuda'),
    #                      'running_mean': weights[f'block{idx}.layer.1.bn2.running_mean'].to('cuda'),
    #                      'running_var': weights[f'block{idx}.layer.1.bn2.running_var'].to('cuda')})
    # output = bn2(output)
    output = F.batch_norm(
        output,
        running_mean=weights_nograd[f"block{idx}.layer.1.bn2.running_mean"],
        running_var=weights_nograd[f"block{idx}.layer.1.bn2.running_var"],
        weight=weights[f"block{idx}.layer.1.bn2.weight"],
        bias=weights[f"block{idx}.layer.1.bn2.bias"],
        eps=1e-05,
        momentum=0.1,
    )
    output = F.relu(output, inplace=True)
    output = F.conv2d(
        output,
        weight=weights[f"block{idx}.layer.1.conv2.weight"],
        bias=None,
        stride=(1, 1),
        padding=(1, 1),
    )
    # output = F.dropout(output, p=0, inplace=False)
    return torch.add(output, input)
