from einops import rearrange, repeat
import torch
from einops.layers.torch import Rearrange
import torch.nn.functional as F


def Attention(weights, x, dim, idx, heads=8, dim_head=64, dropout=0.0):
    scale = dim_head**-0.5
    project_out = not (heads == 1 and dim_head == dim)

    qkv = F.linear(
        x,
        weight=weights[f"transformer.layers.{idx}.0.fn.to_qkv.weight"],
    ).chunk(3, dim=-1)
    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=heads), qkv)

    dots = torch.matmul(q, k.transpose(-1, -2)) * scale
    attn = F.softmax(dots, dim=-1)
    out = torch.matmul(attn, v)
    out = rearrange(out, "b h n d -> b n (h d)")
    if project_out:
        out = F.linear(
            out,
            weight=weights[f"transformer.layers.{idx}.0.fn.to_out.0.weight"],
            bias=weights[f"transformer.layers.{idx}.0.fn.to_out.0.bias"],
        )
        out = F.dropout(out, p=dropout, training=False)

    return out


def FeedForward(weights, x, idx, dropout=0.0):
    # print("FeedForward input:", x[0][0][0])  # Debugging line
    out = F.linear(
        x,
        weight=weights[f"transformer.layers.{idx}.1.fn.net.0.weight"],
        bias=weights[f"transformer.layers.{idx}.1.fn.net.0.bias"],
    )
    out = F.gelu(out)
    out = F.dropout(out, p=dropout)
    # # print("FeedForward input:", out[0][0][0])  # Debugging line

    out = F.linear(
        out,
        weight=weights[f"transformer.layers.{idx}.1.fn.net.3.weight"],
        bias=weights[f"transformer.layers.{idx}.1.fn.net.3.bias"],
    )
    out = F.dropout(out, p=dropout)

    # print("FeedForward output:", out[0][0][0])  # Debugging line
    return out


def transformer(
    weights, weights_no_grad, x, dim, depth, heads, dim_head, mlp_dim, dropout=0.0
):
    for i in range(depth):
        out1 = F.layer_norm(
            x,
            weight=weights[f"transformer.layers.{i}.0.norm.weight"],
            bias=weights[f"transformer.layers.{i}.0.norm.bias"],
            normalized_shape=(dim,),
        )
        out1 = Attention(weights, out1, dim, i, heads, dim_head, dropout)

        out1 = out1 + x
        # # print("out1:", out1[0][0][0])

        out2 = F.layer_norm(
            out1,
            weight=weights[f"transformer.layers.{i}.1.norm.weight"],
            bias=weights[f"transformer.layers.{i}.1.norm.bias"],
            normalized_shape=(dim,),
        )
        out2 = FeedForward(weights, out2, i, dropout)

        x = out2 + out1
        # # print("out2:", x[0][0][0])

    return x


def functional_vit(
    weights,
    weights_nograd,
    x,
    # image_size,
    patch_size,
    # num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    pool="cls",
    # channels=3,
    dim_head=64,
    dropout=0.0,
    emb_dropout=0.0,
):
    x = rearrange(
        x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
    )
    x = F.linear(
        x,
        weight=weights["to_patch_embedding.1.weight"],
        bias=weights["to_patch_embedding.1.bias"],
    )
    b, n, _ = x.shape
    # # print("0", x[0][0][0])

    cls_token = weights["cls_token"]
    # # print("cls_tokens", cls_token[0][0][0])
    # num_patches = (image_size // patch_size) * (image_size // patch_size)
    pos_embedding = weights["pos_embedding"]
    cls_tokens = repeat(cls_token, "() n d -> b n d", b=b)
    x = torch.cat((cls_tokens, x), dim=1)  # type: ignore
    x += pos_embedding[:, : (n + 1)]  # type: ignore
    F.dropout(x, emb_dropout)

    x = transformer(
        weights,
        weights_nograd,
        x,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=dropout,
    )
    # # print("1", x[0][0][0])

    x = x.mean(dim=1) if pool == "mean" else x[:, 0]

    x = F.layer_norm(
        x,
        weight=weights["mlp_head.0.weight"],
        bias=weights["mlp_head.0.bias"],
        normalized_shape=(dim,),
    )

    return F.linear(
        x,
        weight=weights["mlp_head.1.weight"],
        bias=weights["mlp_head.1.bias"],
    )


# # print(functional_vit(x=torch.randn(1, 3, 32, 32), patch_size=4).shape)
# fun = Rearrange(
#     "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
#     p1=4,
#     p2=4,
# )
# # print(fun(torch.randn(1, 3, 32, 32)).shape)
