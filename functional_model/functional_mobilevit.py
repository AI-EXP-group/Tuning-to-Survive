import torch
import torch.nn.functional as F
from einops import rearrange


def MV2Block(weights, weights_nograd, x, idx, inp, oup, stride=1, expansion=4):
    hidden_dim = int(inp * expansion)
    use_res_connect = stride == 1 and inp == oup
    if expansion == 1:
        out = F.conv2d(
            x,
            weight=weights[f"mv2.{idx}.conv.0.weight"],
            stride=stride,
            padding=1,
            groups=hidden_dim,
        )
        out = F.batch_norm(
            out,
            running_mean=weights_nograd["mv2.{idx}.conv.1.running_mean"],
            running_var=weights_nograd["mv2.{idx}.conv.1.running_var"],
            weight=weights["mv2.{idx}.conv.1.weight"],
            bias=weights["mv2.{idx}.conv.1.bias"],
        )
        out = F.silu(out)

        out = F.conv2d(
            out,
            weight=weights[f"mv2.{idx}.conv.3.weight"],
            stride=1,
            padding=0,
        )
        out = F.batch_norm(
            out,
            running_mean=weights_nograd["mv2.{idx}.conv.4.running_mean"],
            running_var=weights_nograd["mv2.{idx}.conv.4.running_var"],
            weight=weights["mv2.{idx}.conv.4.weight"],
            bias=weights["mv2.{idx}.conv.4.bias"],
        )

    else:
        out = F.conv2d(
            x,
            weight=weights[f"mv2.{idx}.conv.0.weight"],
            stride=1,
            padding=0,
        )
        out = F.batch_norm(
            out,
            running_mean=weights_nograd[f"mv2.{idx}.conv.1.running_mean"],
            running_var=weights_nograd[f"mv2.{idx}.conv.1.running_var"],
            weight=weights[f"mv2.{idx}.conv.1.weight"],
            bias=weights[f"mv2.{idx}.conv.1.bias"],
        )
        out = F.silu(out)
        out = F.conv2d(
            out,
            weight=weights[f"mv2.{idx}.conv.3.weight"],
            stride=stride,
            padding=1,
            groups=hidden_dim,
        )
        out = F.batch_norm(
            out,
            running_mean=weights_nograd[f"mv2.{idx}.conv.4.running_mean"],
            running_var=weights_nograd[f"mv2.{idx}.conv.4.running_var"],
            weight=weights[f"mv2.{idx}.conv.4.weight"],
            bias=weights[f"mv2.{idx}.conv.4.bias"],
        )
        out = F.silu(out)
        out = F.conv2d(
            out,
            weight=weights[f"mv2.{idx}.conv.6.weight"],
            stride=1,
            padding=0,
        )
        out = F.batch_norm(
            out,
            running_mean=weights_nograd[f"mv2.{idx}.conv.7.running_mean"],
            running_var=weights_nograd[f"mv2.{idx}.conv.7.running_var"],
            weight=weights[f"mv2.{idx}.conv.7.weight"],
            bias=weights[f"mv2.{idx}.conv.7.bias"],
        )

    if use_res_connect:
        return out + x
    else:
        return out


# def pre_norm(weights, x, idx, dim):
#     out = F.layer_norm(
#         x,
#         weight=weights[f"mvit.{idx}.norm.weight"],
#         bias=weights[f"mvit.{idx}.norm.bias"],
#         normalized_shape=(dim,),
#     )
#     out = F.linear(
#         out,
#         weight=weights[f"mvit.{idx}.linear.weight"],
#         bias=weights[f"mvit.{idx}.linear.bias"],
#     )
#     return out


def conv_nxn_bn(weights, weights_nograd, x, stride=1, pre=""):
    out = F.conv2d(
        x,
        weight=weights[pre + ".0.weight"],
        stride=stride,
        padding=1,
    )
    out = F.batch_norm(
        out,
        running_mean=weights_nograd[pre + ".1.running_mean"],
        running_var=weights_nograd[pre + ".1.running_var"],
        weight=weights[pre + ".1.weight"],
        bias=weights[pre + ".1.bias"],
    )
    out = F.silu(out)
    return out


def conv_1x1_bn(weights, weights_nograd, x, pre="") -> torch.Tensor:
    out = F.conv2d(
        x,
        weight=weights[pre + ".0.weight"],
        stride=1,
        padding=0,
    )
    out = F.batch_norm(
        out,
        running_mean=weights_nograd[pre + ".1.running_mean"],
        running_var=weights_nograd[pre + ".1.running_var"],
        weight=weights[pre + ".1.weight"],
        bias=weights[pre + ".1.bias"],
    )
    out = F.silu(out)
    return out


def Attention(weights, x, dim, heads=8, dim_head=64, dropout=0.0, pre=""):
    scale = dim_head**-0.5
    project_out = not (heads == 1 and dim_head == dim)

    qkv = F.linear(
        x,
        weight=weights[pre + ".to_qkv.weight"],
    ).chunk(3, dim=-1)
    q, k, v = map(lambda t: rearrange(t, "b p n (h d) -> b p h n d", h=heads), qkv)

    dots = torch.matmul(q, k.transpose(-1, -2)) * scale
    attn = F.softmax(dots, dim=-1)
    out = torch.matmul(attn, v)
    out = rearrange(out, "b p h n d -> b p n (h d)")
    if project_out:
        out = F.linear(
            out,
            weight=weights[pre + ".to_out.0.weight"],
            bias=weights[pre + ".to_out.0.bias"],
        )
        out = F.dropout(out, p=dropout, training=False)

    return out


def FeedForward(weights, x, dropout=0.0, pre=""):
    out = F.linear(
        x,
        weight=weights[pre + ".net.0.weight"],
        bias=weights[pre + ".net.0.bias"],
    )
    out = F.silu(out)
    out = F.dropout(out, p=dropout)

    out = F.linear(
        out,
        weight=weights[pre + ".net.3.weight"],
        bias=weights[pre + ".net.3.bias"],
    )
    out = F.dropout(out, p=dropout)

    return out


def transformer(weights, x, dim, depth, heads, dim_head, dropout=0.0, pre=""):
    for i in range(depth):
        out1 = F.layer_norm(
            x,
            weight=weights[pre + f".{i}.0.norm.weight"],
            bias=weights[pre + f".{i}.0.norm.bias"],
            normalized_shape=(dim,),
        )
        out1 = Attention(
            weights, out1, dim, heads, dim_head, dropout, pre=pre + f".{i}.0.fn"
        )

        out1 = out1 + x
        # print("out1:", out1[0][0][0][0])

        out2 = F.layer_norm(
            out1,
            weight=weights[pre + f".{i}.1.norm.weight"],
            bias=weights[pre + f".{i}.1.norm.bias"],
            normalized_shape=(dim,),
        )
        out2 = FeedForward(weights, out2, pre=pre + f".{i}.1.fn")

        # out2 = F.linear(
        #     out2,
        #     weight=weights[pre + f".{i}.1.norm.weight"],
        #     bias=weights[pre + f".{i}.1.norm.bias"],
        # )
        x = out2 + out1
        # print("out2:", x[0][0][0][0])

    return x


def MobileViTBlock(
    weights, weights_nograd, x, idx, dim, depth, channel, patch_size, mlp_dim
):
    y = x.clone()

    out = conv_nxn_bn(weights, weights_nograd, x, pre=f"mvit.{idx}.conv1")
    out = conv_1x1_bn(
        weights,
        weights_nograd,
        out,
        pre=f"mvit.{idx}.conv2",
    )

    # Transformer block (2 layers)
    _, _, h, w = out.shape
    ph, pw = patch_size
    out = rearrange(
        out,
        "b d (h ph) (w pw) -> b (ph pw) (h w) d",
        ph=ph,
        pw=pw,
    )
    # print("out1:", out[0][0][0][0])
    out = transformer(
        weights, out, dim, depth, 4, 8, pre=f"mvit.{idx}.transformer.layers"
    )
    # print("out2:", out[0][0][0][0])
    out = rearrange(
        out,
        "b (ph pw) (h w) d -> b d (h ph) (w pw)",
        h=h // ph,
        w=w // pw,
        ph=ph,
        pw=pw,
    )

    # Fusion
    out = conv_1x1_bn(weights, weights_nograd, out, pre=f"mvit.{idx}.conv3")
    out = torch.cat((out, y), 1)
    out = conv_nxn_bn(weights, weights_nograd, out, pre=f"mvit.{idx}.conv4")
    return out


def MobileViT(
    weights,
    weights_nograd,
    x,
    image_size,
    dims,
    channels,
    num_classes,
    expansion=4,
    kernel_size=3,
    patch_size=(2, 2),
):
    ih, iw = image_size
    L = [2, 4, 3]

    out = conv_nxn_bn(weights, weights_nograd, x, stride=2, pre="conv1")  # ok
    # print("1:", out[0][0][0][0])
    out = MV2Block(
        weights, weights_nograd, out, 0, channels[0], channels[1], 1, expansion
    )  # ok
    # print("2:", out[0][0][0][0])
    out = MV2Block(
        weights, weights_nograd, out, 1, channels[1], channels[2], 2, expansion
    )  # ok
    # print("3:", out[0][0][0][0])
    out = MV2Block(
        weights, weights_nograd, out, 2, channels[2], channels[3], 1, expansion
    )  # ok
    # print("4:", out[0][0][0][0])
    out = MV2Block(
        weights, weights_nograd, out, 3, channels[2], channels[3], 1, expansion
    )  # ok
    # print("5:", out[0][0][0][0])

    out = MV2Block(
        weights, weights_nograd, out, 4, channels[3], channels[4], 2, expansion
    )  # ok
    # print("6:", out[0][0][0][0])
    out = MobileViTBlock(
        weights,
        weights_nograd,
        out,
        0,
        dims[0],
        L[0],
        channels[5],
        patch_size,
        int(dims[0] * 2),
    )
    # print("7:", out[0][0][0][0])
    out = MV2Block(
        weights, weights_nograd, out, 5, channels[5], channels[6], 2, expansion
    )  # ok
    # print("8:", out[0][0][0][0])
    out = MobileViTBlock(
        weights,
        weights_nograd,
        out,
        1,
        dims[1],
        L[1],
        channels[7],
        patch_size,
        int(dims[0] * 4),
    )
    # print("9:", out[0][0][0][0])
    out = MV2Block(
        weights, weights_nograd, out, 6, channels[7], channels[8], 2, expansion
    )  # ok
    # print("0:", out[0][0][0][0])
    out = MobileViTBlock(
        weights,
        weights_nograd,
        out,
        2,
        dims[2],
        L[2],
        channels[9],
        patch_size,
        int(dims[0] * 4),
    )
    # print("1:", out[0][0][0][0])
    out = conv_1x1_bn(weights, weights_nograd, out, pre="conv2")
    # print("2:", out[0][0][0][0])
    out = F.avg_pool2d(out, kernel_size=ih // 32, stride=1)
    out = F.linear(
        out.view(out.size(0), -1),
        weight=weights["fc.weight"],
    )
    return out


def functional_mobilevit_xxs(weights, weights_nograd, x):
    dim = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT(
        weights,
        weights_nograd,
        x,
        image_size=(64, 64),
        dims=dim,
        channels=channels,
        num_classes=200,
        expansion=2,
    )
    # return MobileViT(
    #     image_size=(224, 224),
    #     dims=dim,
    #     channels=channels,
    #     num_classes=1000,
    #     expansion=4,
    #     kernel_size=3,
    #     patch_size=(2, 2),
    # )


def functional_mobilevit_s(weights, weights_nograd, x):
    dim = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT(
        weights,
        weights_nograd,
        x,
        image_size=(64, 64),
        dims=dim,
        channels=channels,
        num_classes=200,
        expansion=4,
    )
