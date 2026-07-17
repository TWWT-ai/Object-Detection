import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision

architecture_config = [
    # Not including fully connected head
    # (kernel, number of filters, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # Values inside tuple are the same, last index is the number of repeatition
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channel)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=False)

    def forward(self, x):
        return self.leaky_relu(self.batch_norm(self.conv(x)))


class YOLOv1(nn.Module):
    """YOLOv1 head + loss on an ImageNet-PRETRAINED ResNet18 backbone.

    The paper's recipe (section 2.2) pretrains its conv layers on ImageNet
    before detection training — that step is what makes 'classic YOLO detects
    well' true. We substitute ResNet18 pretrained weights for the week-long
    darknet pretraining we cannot reproduce. The FC head, loss and target
    encoding stay exactly as before.
    """

    def __init__(self, in_channels=4, **kwargs):
        super(YOLOv1, self).__init__()

        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )

        # First conv expects 3 channels; ours has 4 (RGB-D). Keep the
        # pretrained RGB filters and initialise the depth channel with their
        # mean, so pretraining is preserved instead of thrown away.
        old = resnet.conv1
        new = nn.Conv2d(in_channels, old.out_channels,
                        kernel_size=7, stride=2, padding=3, bias=False)
        with th.no_grad():
            new.weight[:, :3] = old.weight
            if in_channels > 3:
                new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
        resnet.conv1 = new

        # Everything except avgpool/fc: 448x448 input -> [N, 512, 14, 14]
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Paper-style final convs (randomly initialised, like the paper's
        # detection-specific layers): 14x14 -> 7x7, 512 -> 1024 channels,
        # so _create_fc receives exactly the same shape as before.
        self.neck = nn.Sequential(
            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
        )

        self.fully_connected = self._create_fc(**kwargs)

        # ImageNet models were trained on mean/std-normalised RGB; our
        # dataloader only divides by 255. Normalise here so the pretrained
        # filters see the statistics they were trained on. Depth passes as-is.
        self.register_buffer("rgb_mean", th.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", th.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        rgb = (x[:, :3] - self.rgb_mean) / self.rgb_std
        x = th.cat([rgb, x[:, 3:]], dim=1)
        x = self.neck(self.backbone(x))
        return self.fully_connected(th.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels,
                        out_channel=x[1],
                        kernel_size=x[0],
                        stride=x[2],
                        padding=x[3]
                    )
                ]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeat = x[-1]

                for _ in range(num_repeat):
                    layers += [
                        CNNBlock(
                            in_channels,
                            out_channel=conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3]
                        )
                    ]

                    layers += [
                        CNNBlock(
                            conv1[1],
                            out_channel=conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fc(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),  # For low vram use 496
            nn.Dropout(0.3),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (B * 5 + C))
        )


def test(S=7, B=2, C=20):
    model = YOLOv1(
        in_channels=4,
        split_size=S,
        num_boxes=B,
        num_classes=C
    )
    x = th.randn((2, 4, 448, 448))
    print(model(x).shape)


# Guarded: without this, every `import model` (train/evaluate/visualize)
# builds a throwaway 200M-param model and runs a CPU forward pass
if __name__ == "__main__":
    test()
