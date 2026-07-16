import torch as th
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_channels=4, **kwargs):
        super(YOLOv1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fully_connected = self._create_fc(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
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
