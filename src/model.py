import torch as th
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels, kernel_size=3, stride=1):
    """Convolution, batch normalisation, and LeakyReLU used by every encoder block."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True),
    )


class Backbone(nn.Module):
    """Shared RGB-D encoder. A 448 x 448 image becomes a 7 x 7 feature map."""

    def __init__(self, in_channels=4):
        super().__init__()
        self.stage1 = nn.Sequential(conv_block(in_channels, 32, stride=2), conv_block(32, 32))
        self.stage2 = nn.Sequential(conv_block(32, 64, stride=2), conv_block(64, 64))
        self.stage3 = nn.Sequential(conv_block(64, 128, stride=2), conv_block(128, 128))
        self.stage4 = nn.Sequential(conv_block(128, 256, stride=2), conv_block(256, 256))
        self.stage5 = nn.Sequential(conv_block(256, 512, stride=2), conv_block(512, 512))
        self.stage6 = nn.Sequential(conv_block(512, 1024, stride=2), conv_block(1024, 1024))

    def forward(self, x):
        f224 = self.stage1(x)
        f112 = self.stage2(f224)
        f56 = self.stage3(f112)
        f28 = self.stage4(f56)
        f14 = self.stage5(f28)
        f7 = self.stage6(f14)
        return f56, f28, f14, f7


class DetectionHead(nn.Module):
    """Returns B candidate boxes per cell in [N, 7, 7, B * 5] layout."""

    def __init__(self, num_boxes=2):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(1024, 512),
            nn.Conv2d(512, num_boxes * 5, kernel_size=1),
        )

    def forward(self, f7):
        return self.net(f7).permute(0, 2, 3, 1).contiguous()


class ClassificationHead(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, f7):
        average_features = F.adaptive_avg_pool2d(f7, 1).flatten(1)
        maximum_features = F.adaptive_max_pool2d(f7, 1).flatten(1)
        return self.net(th.cat([average_features, maximum_features], dim=1))


class SegmentationHead(nn.Module):
    """Decoder with encoder skip connections; output is an unnormalised mask logit."""

    def __init__(self):
        super().__init__()
        self.up1 = conv_block(1024 + 512, 256)
        self.up2 = conv_block(256 + 256, 128)
        self.up3 = conv_block(128 + 128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, f56, f28, f14, f7):
        x = F.interpolate(f7, size=f14.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(th.cat([x, f14], dim=1))
        x = F.interpolate(x, size=f28.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(th.cat([x, f28], dim=1))
        x = F.interpolate(x, size=f56.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(th.cat([x, f56], dim=1))
        x = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)
        return self.out(x)


class HandGestureNet(nn.Module):
    """Shared RGB-D network for detection, segmentation, and gesture classification."""

    def __init__(self, in_channels=4, n_classes=10, B=2):
        super().__init__()
        self.backbone = Backbone(in_channels)
        self.detection_head = DetectionHead(B)
        self.segmentation_head = SegmentationHead()
        self.classification_head = ClassificationHead(n_classes)

    def forward(self, x):
        f56, f28, f14, f7 = self.backbone(x)
        return (
            self.detection_head(f7),
            self.segmentation_head(f56, f28, f14, f7),
            self.classification_head(f7),
        )
