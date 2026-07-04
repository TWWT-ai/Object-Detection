import torch as th
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, B=2):
        super().__init__()
        self.B = B
        self.net = nn.Sequential(
            # Reforming 1024 of data into 512 constructed with newly mixed weights with lower weights become negligible
            conv_block(1024, 512), 
            # To determine the frame that actually lands on what we want
            # Candidate box(possible box containing the object) * (x, y, w, h, conf) = B * 5
            nn.Conv2d(512, B * 5, kernel_size=1) 
        )


    def forward(self, f7):
        output = self.net(f7)
        # Convert back to the form which torch expects (batch, channel, height, width), (batch, 7, 7, 10)
        return output.permute(0, 2, 3, 1)   # Number is the index


class ClassificationHead(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        # To classify which sign is the possible class we want (returns logit)
        self.fc = nn.Linear(1024, n_classes)


    def forward(self, f7):
        # Extracting from each block whether is contains the object
        v = F.adaptive_avg_pool2d(f7, 1).flatten(1)
        return self.fc(v)
    
    
class SegmentationHead(nn.Module):
    # Resizing back to input image size with suitable correction and modification
    def __init__(self):
        super().__init__()
        # Sticking back the channels
        self.up1 = conv_block(1024 + 512, 256)
        self.up2 = conv_block(256 + 256, 128)
        self.up3 = conv_block(128 + 128, 64)
        # Converting the all 64 weight per pixel into one number
        self.out = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, f56, f28, f14, f7):
        # Enlarging by scale factor 2
        x = F.interpolate(f7, scale_factor=2)
        # Following the channel(dimension 1) extend on it e.g. 1025 + 512 = 1536
        x = self.up1(th.cat([x, f14], dim=1))
        x = F.interpolate(x, scale_factor=2)
        x = self.up2(th.cat([x, f28], dim=1))
        x = F.interpolate(x, scale_factor=2)
        x = self.up3(th.cat([x, f56], dim=1))
        x = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)
        return self.out(x)
    

class HandGestureNet(nn.Module):
    def __init__(self):
        pass
