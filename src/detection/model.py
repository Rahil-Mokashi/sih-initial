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


def build_model(encoder_name: str = "resnet18", encoder_weights: str | None = "imagenet", in_channels: int = 1) -> nn.Module:
    """Binary segmentation U-Net, classes=1 (oil vs. not-oil).

    in_channels defaults to 1 (single-band SAR) to match the production
    checkpoint the dashboard/inference scripts load. The Zenodo GeoTIFFs
    actually carry 2 real bands (confirmed by direct inspection -- distinct
    dB ranges per band, not a duplicate), so in_channels=2 is a real,
    separately-tracked option for scripts/train_detection.py's --channels
    flag -- NOT the default here, since the real PANGAEA case-study images
    used by the live dashboard (ow-0001 etc.) are single-band JPGs and can't
    supply a second channel. smp adapts its first conv layer for
    in_channels != 3 automatically (still uses ImageNet-pretrained weights
    for the rest of the encoder).
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1,
    )
