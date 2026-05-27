import torch
import torch.nn.functional as F


def embeddings_forward(weights, x, patch_size: int):
    cls_token = weights["embeddings.cls_token"]
    position_embeddings = weights["embeddings.position_embeddings"]

    out = F.conv2d(
        x,
        weight=weights["embeddings.projection.weight"],
        bias=weights["embeddings.projection.bias"],
        stride=(patch_size, patch_size),
        padding=(0, 0),
    )

    out = out.flatten(2).transpose(1, 2)  # [B, num_patches, D]
    out = torch.cat([cls_token.repeat(x.size(0), 1, 1), out], dim=1)
    out = out + position_embeddings  # [2, 145, 108]

    return out


def multi_head_attention(weights, x, idx, D: int, H: int, dropout: float):
    # dropout = dropout
    d_k = D // H
    B, N, _ = x.size()

    q = F.linear(
        x,
        weight=weights[f"encoder.{idx}.msa.query.weight"],
        bias=weights[f"encoder.{idx}.msa.query.bias"],
    ).view(B, H, N, d_k)
    # print(q[0][0][0][0])
    k = F.linear(
        x,
        weight=weights[f"encoder.{idx}.msa.key.weight"],
        bias=weights[f"encoder.{idx}.msa.key.bias"],
    ).view(B, H, N, d_k)
    # print(k[0][0][0][0])
    v = F.linear(
        x,
        weight=weights[f"encoder.{idx}.msa.value.weight"],
        bias=weights[f"encoder.{idx}.msa.value.bias"],
    ).view(B, H, N, d_k)
    # print(v[0][0][0][0])
    dots = (q @ k.transpose(2, 3)) / (d_k**0.5)  # [2, 6, 145, 145]
    # print(dots[0][0][0][0])
    attn = F.softmax(dots, dim=3)  # [2, 6, 145, 145]
    # print(attn[0][0][0][0])

    out = attn @ v  # [2, 6, 145, 18]
    out = out.transpose(1, 2).reshape(B, N, D)  # [2, 145, 108]
    # print(out[0][0][0])
    out = F.linear(
        out,
        weight=weights[f"encoder.{idx}.msa.output.weight"],
        bias=weights[f"encoder.{idx}.msa.output.bias"],
    )
    # print(out[0][0][0])
    return out  # [2, 145, 108]


def encoder_forward(weights, x, idx, D: int, H: int, dropout: float):
    # print(x[0][0][0])
    residual = x
    x = F.layer_norm(
        x,
        weight=weights[f"encoder.{idx}.layernorm_before.weight"],
        bias=weights[f"encoder.{idx}.layernorm_before.bias"],
        normalized_shape=(D,),
    )
    # print(x[0][0][0])
    x = multi_head_attention(weights, x, idx, D, H, dropout)
    # print(x[0][0][0])
    x = residual + x

    residual = x

    x = F.layer_norm(
        x,
        weight=weights[f"encoder.{idx}.layernorm_after.weight"],
        bias=weights[f"encoder.{idx}.layernorm_after.bias"],
        normalized_shape=(D,),
    )

    x = F.gelu(
        F.linear(
            x,
            weight=weights[f"encoder.{idx}.intermediate.weight"],
            bias=weights[f"encoder.{idx}.intermediate.bias"],
        )
    )
    x = F.linear(
        x,
        weight=weights[f"encoder.{idx}.output.weight"],
        bias=weights[f"encoder.{idx}.output.bias"],
    )
    x = residual + x
    return x


def functional_vit(
    weights,
    x,
    blocks: int = 6,
    channels: int = 3,
    patch_size: int = 16,
    H: int = 12,
    dropout: float = 0,
):

    x = embeddings_forward(weights, x, patch_size)

    D = (patch_size**2) * channels
    for idx in range(0, blocks):
        x = encoder_forward(weights, x, idx, D, H, dropout)
    # print(x)
    x = F.layer_norm(
        x,
        weight=weights["layernorm.weight"],
        bias=weights["layernorm.bias"],
        normalized_shape=(D,),
    )

    pooler = F.linear(x, weight=weights["pooler.weight"], bias=weights["pooler.bias"])

    cls_token = pooler[:, 0]

    out = F.layer_norm(
        cls_token,
        weight=weights["mlp_head.0.weight"],
        bias=weights["mlp_head.0.bias"],
        normalized_shape=(D,),
    )

    mlp_head = F.linear(
        out, weight=weights["mlp_head.1.weight"], bias=weights["mlp_head.1.bias"]
    )
    return mlp_head


# from networks import  vit_b_16
# from watermark_utils import *


# model = vit_b_16.Vit(classes=200, blocks=12, channels=3, height=224, width=224,
#             patch_size=16,
#             H=12, inner_dim=3072, dropout=0.1)
# model.eval()

# tensor = torch.rand([2, 3, 224, 224])
# weight_s, _ = get_weights(model)

# with torch.no_grad():
#     print(model(tensor)[0][0])
#     print()
#     print(functional_vit(weight_s, tensor, blocks=12, channels=3,
#             patch_size=16,
#             H=12, dropout=0.1)[0][0])
