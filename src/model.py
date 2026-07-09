import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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
        # >>> PRETRAINED CHANGE: use ImageNet-pretrained ResNet34 instead of training from scratch
        resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)

        # >>> PRETRAINED CHANGE: inflate first conv 3ch -> in_channels(4).
        # Keep pretrained RGB weights; init the depth channel from the mean of RGB weights.
        old = resnet.conv1                                   # Conv2d(3, 64, 7, stride=2, padding=3)
        new = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with th.no_grad():
            new.weight[:, :3] = old.weight
            if in_channels > 3:
                new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)
        resnet.conv1 = new

        # 448 -> 112 (conv1 s2, then maxpool s2)
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1     # 112, 64ch
        self.layer2 = resnet.layer2     # 56,  128ch -> f56
        self.layer3 = resnet.layer3     # 28,  256ch -> f28
        self.layer4 = resnet.layer4     # 14,  512ch -> f14
        # >>> PRETRAINED CHANGE: one extra stride-2 block to reach the 7x7 / 1024ch that the heads expect
        self.to_f7  = conv_block(512, 1024, s=2)   # 14 -> 7

    def forward(self, x):
        x   = self.stem(x)        # 112
        x   = self.layer1(x)      # 112
        f56 = self.layer2(x)      # 56,  128
        f28 = self.layer3(f56)    # 28,  256
        f14 = self.layer4(f28)    # 14,  512
        f7  = self.to_f7(f14)     # 7,   1024
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
        self.net = nn.Sequential(
            nn.Linear(2048, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )


    def forward(self, f7):
        # Extracting from each block whether is contains the object
        avg = F.adaptive_avg_pool2d(f7, 1).flatten(1)
        mx = F.adaptive_max_pool2d(f7, 1).flatten(1)
        return self.net(th.cat([avg, mx], dim=1))
    
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
        # Following the channel(dimension 1) extend on it e.g. 1024 + 512 = 1536
        x = self.up1(th.cat([x, f14], dim=1))
        x = F.interpolate(x, scale_factor=2)
        x = self.up2(th.cat([x, f28], dim=1))
        x = F.interpolate(x, scale_factor=2)
        x = self.up3(th.cat([x, f56], dim=1))
        x = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)
        return self.out(x)
    

class HandGestureNet(nn.Module):
    def __init__(self, in_channels=5, n_classes=10, B=2):
        super().__init__()
        self.backbone = Backbone(in_channels)
        self.detecting_head = DetectionHead(B)
        self.classification_head = ClassificationHead(n_classes)
        self.segmentation_head = SegmentationHead()

    
    def forward(self, x):
        f56, f28, f14, f7 = self.backbone(x)
        return (self.detecting_head(f7),
                self.segmentation_head(f56, f28, f14, f7),
                self.classification_head(f7)
                )
