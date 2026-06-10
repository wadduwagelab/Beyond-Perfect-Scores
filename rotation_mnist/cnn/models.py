"""
LightCNN model ("SimpleCNN_MNIST") for the Rotated-MNIST permutation test.

1-channel 28x28 MNIST input -> binary (treated vs. untreated) logits.
"""

import torch
import torch.nn as nn


class SimpleCNN_MNIST(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN_MNIST, self).__init__()
        # MNIST has 1 channel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # 28x28 -> 14x14 -> 7x7 -> 3x3 after 3 poolings
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
