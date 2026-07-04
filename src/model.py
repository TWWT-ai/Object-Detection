import torch as th
import torch.nn as nn

def conv_block(channel_in, channel_out, k=3, s=1):
    # s=1 only thickening, s=2 shrinking and thickening
    return nn.Sequential(
        # Converting the picture into thinkened and shrinked picture
        nn.Conv2d(channel_in, channel_out, k, stride=s, padding=(k // 2), bias=False),
        nn.BatchNorm2d(channel_out),
        nn.LeakyReLU(0.1, inplace=True)
    )


class Backbone(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        # Reducing the pixels into 7x7 (448 / 2^6)
        # 7x7 being the box of determining the size of the grid cell
        self.s1 = nn.Sequential(conv_block(in_channels, 32, s=2), conv_block(32, 32))
        self.s2 = nn.Sequential(conv_block(32, 64, s=2), conv_block(64, 64))
        self.s3 = nn.Sequential(conv_block(64, 128, s=2), conv_block(128, 128))
        self.s4 = nn.Sequential(conv_block(128, 256, s=2), conv_block(256, 256))
        self.s5 = nn.Sequential(conv_block(256, 512, s=2), conv_block(512, 512))
        self.s6 = nn.Sequential(conv_block(512, 1024, s=2), conv_block(1024, 1024))

    def forward(self, x):
        # Forward pass to get data from size 7 to 56 and later hand to SegmentationHead to help to classify the border
        # Mainly using s3 to s5 to calibrate the shape, therefore keeping the record all these data and ready to share backbone
        f56 = self.s3(self.s2(self.s1(x)))
        f28 = self.s4(f56)
        f14 = self.s5(f28)
        f7 = self.s6(f14)
        return f56, f28, f14, f7



class DetectionHead(nn.Module):
    def __init__(self):
        pass
class SegmentationHead(nn.Module):
    def __init__(self):
        pass
class HandGestureNet(nn.Module):
    def __init__(self):
        pass
