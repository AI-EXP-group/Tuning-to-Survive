#!/usr/bin/env python3
"""
Alignment test: verify that functional models in functional_model/ produce the same
output as the original models in networks/.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.watermark_utils import get_weights

# ============================================================
# Import original network models
# ============================================================
from networks.densenet import densenet_cifar
from networks.googlenet import googlenet
from networks.mobilevit import mobilevit_xxs
from networks.resnet import ResNet, BasicBlock
from networks.vit import ViT
from networks.vit_b_16 import Vit as VitB16
from networks.vit_s import ViT as ViTS
from networks.wresnet import WideResNet

# ============================================================
# Import functional models (note: handle name conflicts between vit_b_16 / vit)
# ============================================================
from functional_model.functional_dense121 import functional_dense121
from functional_model.functional_googlenet import functional_googlenet
from functional_model.functional_mobilevit import functional_mobilevit_xxs
from functional_model.functional_res18 import functional_res18
from functional_model.functional_vit import functional_vit
from functional_model.functional_vit_b_16 import functional_vit as functional_vit_b16
from functional_model.functional_vit_s import functional_vit_s
from functional_model.functional_wrn import functional_wrn


def test_alignment(name, model, func_forward, x, atol=1e-5, **func_kwargs):
    """
    Compare the output of the original model with the functional model.

    Args:
        name: model name (for printing)
        model: original nn.Module model
        func_forward: functional forward function
        x: input tensor (B, C, H, W)
        atol: absolute tolerance
        **func_kwargs: extra keyword arguments passed to the functional function
    """
    model.eval()
    with torch.no_grad():
        out_orig = model(x)

    weights, weights_nograd = get_weights(model)

    with torch.no_grad():
        # Functional function signatures may be func(weights, weights_nograd, x, ...)
        # or func(weights, x, ...) like vit_b_16
        out_func = func_forward(weights, weights_nograd, x, **func_kwargs)

    diff = (out_orig - out_func).abs().max().item()
    status = "✅ PASS" if diff < atol else "❌ FAIL"
    print(f"[{status}] {name:20s} | max diff: {diff:.2e} | atol: {atol}")

    if diff >= atol:
        print(f"  orig[0][:5]:  {out_orig[0][:5]}")
        print(f"  func[0][:5]:  {out_func[0][:5]}")

    return diff < atol


def test_alignment_nograd(name, model, func_forward, x, atol=1e-5, **func_kwargs):
    """
    Variant for functional functions without weights_nograd parameter
    (e.g. functional_vit_b_16).  Signature: func_forward(weights, x, ...)
    """
    model.eval()
    with torch.no_grad():
        out_orig = model(x)

    weights, _ = get_weights(model)

    with torch.no_grad():
        out_func = func_forward(weights, x, **func_kwargs)

    diff = (out_orig - out_func).abs().max().item()
    status = "✅ PASS" if diff < atol else "❌ FAIL"
    print(f"[{status}] {name:20s} | max diff: {diff:.2e} | atol: {atol}")

    if diff >= atol:
        print(f"  orig[0][:5]:  {out_orig[0][:5]}")
        print(f"  func[0][:5]:  {out_func[0][:5]}")

    return diff < atol


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    print("=" * 70)

    results = []

    # ----------------------------------------------------------
    # 1. DenseNet-121 (CIFAR)
    # ----------------------------------------------------------
    model = densenet_cifar(num_classes=10).to(device)
    x = torch.randn(2, 3, 32, 32).to(device)
    results.append(
        test_alignment("DenseNet-121", model, functional_dense121, x, atol=1e-4)
    )

    # ----------------------------------------------------------
    # 2. GoogleNet (CIFAR)
    # ----------------------------------------------------------
    model = googlenet().to(device)
    x = torch.randn(2, 3, 32, 32).to(device)
    results.append(
        test_alignment("GoogleNet", model, functional_googlenet, x, atol=1e-4)
    )

    # ----------------------------------------------------------
    # 3. MobileViT-XXS
    # ----------------------------------------------------------
    model = mobilevit_xxs().to(device)
    x = torch.randn(2, 3, 64, 64).to(device)
    results.append(
        test_alignment("MobileViT-XXS", model, functional_mobilevit_xxs, x, atol=1e-4)
    )

    # ----------------------------------------------------------
    # 4. ResNet-18 (CIFAR)
    # ----------------------------------------------------------
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
    x = torch.randn(2, 3, 32, 32).to(device)
    results.append(test_alignment("ResNet-18", model, functional_res18, x, atol=1e-4))

    # ----------------------------------------------------------
    # 5. ViT (vit.py) - basic ViT for small datasets
    # ----------------------------------------------------------
    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=256,
        dropout=0.0,
        emb_dropout=0.0,
    ).to(device)
    x = torch.randn(2, 3, 32, 32).to(device)
    results.append(
        test_alignment(
            "ViT (vit.py)",
            model,
            functional_vit,
            x,
            atol=1e-4,
            patch_size=4,
            dim=256,
            depth=6,
            heads=8,
            mlp_dim=256,
        )
    )

    # ----------------------------------------------------------
    # 6. ViT-B/16 (vit_b_16.py) - 注意: functional_vit_b16(weights, x, ...)
    #    no weights_nograd parameter
    # ----------------------------------------------------------
    model = VitB16(
        classes=10,
        blocks=6,
        channels=3,
        height=224,
        width=224,
        patch_size=16,
        H=6,
        inner_dim=1536,
        dropout=0.1,
    ).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    results.append(
        test_alignment_nograd(
            "ViT-B/16",
            model,
            functional_vit_b16,
            x,
            atol=1e-4,
            blocks=6,
            channels=3,
            patch_size=16,
            H=6,
            dropout=0.1,
        )
    )

    # ----------------------------------------------------------
    # 7. ViT-S (vit_s.py) - ViT for small datasets with SPT & LSA
    # ----------------------------------------------------------
    model = ViTS(
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.0,
        emb_dropout=0.0,
    ).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    results.append(
        test_alignment(
            "ViT-S",
            model,
            functional_vit_s,
            x,
            atol=1e-4,
            patch_size=16,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=512,
        )
    )

    # ----------------------------------------------------------
    # 8. WideResNet (WRN-16-4)
    # ----------------------------------------------------------
    model = WideResNet(depth=16, num_classes=10, widen_factor=4).to(device)
    x = torch.randn(2, 3, 32, 32).to(device)
    results.append(test_alignment("WRN-16-4", model, functional_wrn, x, atol=1e-4))

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"🎉 All {total}/{total} models passed!")
    else:
        print(f"⚠️  {passed}/{total} models passed, {total - passed} failed.")


if __name__ == "__main__":
    main()
