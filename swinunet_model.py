# swinunet_model.py
"""
Lightweight Swin‑UNet hybrid for brain‑tumor detection.
Encoder: Swin‑Tiny (pre‑trained on ImageNet, first 3 stages)
Decoder: shallow UNet‑style up‑sampling
Outputs:
    • seg  – low‑res segmentation map  (B,1,H/4,W/4)
    • prob – classification probability (B,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # PyTorch Image Models  (pip install timm)

# --------------------------------------------------
# Encoder
# --------------------------------------------------
class SwinEncoder(nn.Module):
    """
    Swin‑Tiny encoder. timm returns NHWC (channels_last) tensors, so we
    permute to NCHW before feeding to the decoder.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2)   # 0:96c, 1:192c, 2:384c
        )
        self.out_channels = [96, 192, 384]

    def forward(self, x):
        feats = self.backbone(x)  # list of NHWC tensors
        # convert to NCHW
        feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
        return feats  # f1, f2, f3 (96, 192, 384)

# --------------------------------------------------
# Simple up‑sampling block  (UNet style)
# --------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# --------------------------------------------------
# Swin‑UNet model
# --------------------------------------------------
class SwinUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder
        self.enc = SwinEncoder(pretrained=True)
        c1, c2, c3 = self.enc.out_channels   # 96, 192, 384

        # decoder
        self.up2 = UpBlock(c3, c2, c2)   # 384→192
        self.up1 = UpBlock(c2, c1, c1)   # 192→96

        # segmentation head (low‑res map)
        self.seg_head = nn.Conv2d(c1, 1, kernel_size=1)

        # classification head (global avg pool on deepest feat)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, 1)
        )

    def forward(self, x):
        f1, f2, f3 = self.enc(x)

        d2 = self.up2(f3, f2)         # 384→192
        d1 = self.up1(d2, f1)         # 192→96

        seg = torch.sigmoid(self.seg_head(d1))            # (B,1,H/4,W/4)
        cls = torch.sigmoid(self.cls_head(f3)).squeeze(1) # (B,)
        return seg, cls
