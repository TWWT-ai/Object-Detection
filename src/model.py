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


def res_stage(in_ch, out_ch):
    return nn.Sequential(
        Residual(in_ch, out_ch, use_1x1conv=True, strides=2),
        Residual(out_ch, out_ch),
    )


class Residual(nn.Module):
    def __init__(self, input_channel, output_channel, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.batch_norm1 = nn.BatchNorm2d(output_channel)
        self.batch_norm2 = nn.BatchNorm2d(output_channel)
        self.leaky_relu = nn.LeakyReLU(inplace=True)


    def forward(self, X):
        Y = F.leaky_relu(self.batch_norm1(self.conv1(X)), 0.1)
        Y = self.batch_norm2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.leaky_relu(Y)


class Backbone(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        # Reducing the pixels into 7x7 (448 / 2^6)
        # 7x7 being the box of determining the size of the grid cell
        self.s1 = res_stage(in_channels, 32)
        self.s2 = res_stage(32,  64)
        self.s3 = res_stage(64,  128)
        self.s4 = res_stage(128, 256)
        self.s5 = res_stage(256, 512)
        self.s6 = res_stage(512, 1024)


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
