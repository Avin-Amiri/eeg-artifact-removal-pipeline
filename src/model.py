import torch
import torch.nn as nn


class EEGNetLight(nn.Module):

    def __init__(self, num_channels=32, num_classes=2, samples=128):
        super(EEGNetLight, self).__init__()

        # Block 1: Temporal + Spatial Depthwise Conv
        self.conv1 = nn.Conv2d(
            1, 8, (1, 32), padding=(0, 16), bias=False
        )  # Temporal
        self.bn1 = nn.BatchNorm2d(8)

        self.depthwise = nn.Conv2d(
            8, 16, (num_channels, 1), groups=8, bias=False
        )  # Spatial
        self.bn2 = nn.BatchNorm2d(16)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout = nn.Dropout(0.25)

        # Block 2: Separable Conv
        self.separable = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 8))

        out_dim = (samples // 32) * 16
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(out_dim, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.separable(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        return self.classifier(x)
