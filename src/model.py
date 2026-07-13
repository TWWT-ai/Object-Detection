"""YOLOv1-style multi-task network: detection head + U-Net-style segmentation head.

Why not the original Darknet-24 backbone?  The YOLOv1 paper pretrains Darknet-24 on
ImageNet for a WEEK before detection training.  With ~60 training images we would
badly overfit a from-scratch backbone, so we keep the YOLOv1 *detection formulation*
(SxS grid, B boxes/cell, C class scores/cell, same loss) but put it on top of an
ImageNet-pretrained ResNet18 — small, fast on Colab, and its transferred features are
exactly what a tiny dataset needs.

Why one shared backbone for detection AND segmentation?  Both tasks need the same
low/mid-level hand features; sharing them halves compute and acts as mutual
regularisation (the seg loss shapes features that also help detection, and vice versa).
"""

import torch                                   # tensor ops
import torch.nn as nn                          # layers
import torch.nn.functional as F                # interpolate (upsampling)
from torchvision.models import resnet18, ResNet18_Weights  # pretrained backbone

from . import config                           # S, B, C, IMG_SIZE


def conv_block(in_ch, out_ch):
    """3x3 conv + BatchNorm + LeakyReLU — the standard YOLO building block.

    LeakyReLU(0.1) instead of ReLU because that is what YOLOv1 used: it keeps a small
    gradient for negative activations, which helps the regression outputs early on.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),  # bias=False: BN has its own shift
        nn.BatchNorm2d(out_ch),                              # BN stabilises training on small batches
        nn.LeakyReLU(0.1, inplace=True),                     # paper's activation
    )


class YoloV1Gesture(nn.Module):
    """Backbone (ResNet18) -> detection head (7x7 grid) + segmentation head (full-res mask)."""

    def __init__(self):
        super().__init__()
        res = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)   # load ImageNet weights (crucial: tiny dataset)
        # split the resnet into stages so we can tap intermediate features for the seg decoder
        self.stem = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)  # output stride /4
        self.layer1 = res.layer1      # /4,  64 channels  — fine details for the seg decoder
        self.layer2 = res.layer2      # /8, 128 channels
        self.layer3 = res.layer3      # /16, 256 channels
        self.layer4 = res.layer4      # /32, 512 channels — semantic features for detection

        out_ch = config.B * 5 + config.C                         # B boxes * (x,y,w,h,conf) + C classes
        self.det_head = nn.Sequential(                           # detection head on the /32 map
            nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False),  # stride 2: 14x14 -> 7x7 = SxS grid
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            conv_block(512, 256),                                # extra capacity, like YOLO's final convs
            nn.Conv2d(256, out_ch, 1),                           # 1x1 conv -> per-cell prediction vector
        )

        # U-Net-style decoder: repeatedly upsample and fuse with the matching encoder stage.
        # Skip connections are why U-Net beats plain upsampling: they restore the spatial
        # detail that the encoder's downsampling threw away.
        self.up3 = conv_block(512 + 256, 256)                    # fuse /32 (upsampled) with /16
        self.up2 = conv_block(256 + 128, 128)                    # fuse with /8
        self.up1 = conv_block(128 + 64, 64)                      # fuse with /4
        self.seg_out = nn.Conv2d(64, 1, 1)                       # 1 channel: binary hand mask logits

    def forward(self, x):
        x = self.stem(x)                                         # /4
        f1 = self.layer1(x)                                      # /4,  64ch (skip connection 1)
        f2 = self.layer2(f1)                                     # /8, 128ch (skip connection 2)
        f3 = self.layer3(f2)                                     # /16,256ch (skip connection 3)
        f4 = self.layer4(f3)                                     # /32,512ch (deepest features)

        det = self.det_head(f4)                                  # (N, B*5+C, 7, 7)
        det = torch.sigmoid(det)                                 # sigmoid bounds every output to (0,1):
        #                                                          x,y are cell-relative in [0,1], w,h are
        #                                                          image-relative in [0,1], conf and class
        #                                                          scores are probabilities. The original
        #                                                          paper used unbounded linear outputs, but
        #                                                          sigmoid makes training on a tiny dataset
        #                                                          far more stable (no exploding boxes).
        det = det.permute(0, 2, 3, 1)                            # -> (N, S, S, B*5+C): loss-friendly layout

        d3 = self.up3(torch.cat([F.interpolate(f4, size=f3.shape[2:], mode='bilinear',
                                               align_corners=False), f3], dim=1))  # /16 fused
        d2 = self.up2(torch.cat([F.interpolate(d3, size=f2.shape[2:], mode='bilinear',
                                               align_corners=False), f2], dim=1))  # /8 fused
        d1 = self.up1(torch.cat([F.interpolate(d2, size=f1.shape[2:], mode='bilinear',
                                               align_corners=False), f1], dim=1))  # /4 fused
        seg = F.interpolate(self.seg_out(d1), size=(config.IMG_SIZE, config.IMG_SIZE),
                            mode='bilinear', align_corners=False)  # logits upsampled to full 448x448
        return det, seg                                          # detection grid + mask logits