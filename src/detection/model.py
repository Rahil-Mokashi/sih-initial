"""
Detection model: U-Net with a ResNet18 encoder for binary oil segmentation.

Architecture choice (see DECISIONS.md "Step 1: detection model
architecture" for the full writeup): U-Net decoder + ResNet18 encoder,
built with `segmentation_models_pytorch`, chosen over ResNet34/EfficientNet-B0
for VRAM headroom on the 6GB RTX 4050 -- ResNet18 has the fewest parameters
and the simplest (plain conv, no bottleneck/depthwise) blocks of the three,
which keeps activation memory predictable at training time. DeepLabV3+ was
considered but its atrous spatial pyramid pooling head holds several
parallel dilated-conv feature maps in memory simultaneously, which is a
worse fit for a 6GB budget than U-Net's single skip-connection path for the
same encoder.
"""

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(encoder_name: str = "resnet18", encoder_weights: str | None = "imagenet") -> nn.Module:
    """Binary segmentation U-Net. in_channels=1 for single-band SAR input, classes=1 (oil vs. not-oil)."""
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=1,
        classes=1,
    )
