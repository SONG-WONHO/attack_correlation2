from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()
        self.conv1 = nn.Conv2d(1,4, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(4, 12, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(588, 10)
    def forward(self, x, index=-1, metric=0):
        layer = F.relu(self.conv1(x))
        layer = F.max_pool2d(layer, 2)
        layer = F.relu(self.conv2(layer))
        layer = F.max_pool2d(layer, 2)

        layer = layer.view(-1, 588)
        layer = self.fc1(layer)
        output = F.log_softmax(layer, dim=1)
        return output

class LeNet3(nn.Module):
    def __init__(self):
        super(LeNet3, self).__init__()
        self.conv1 = nn.Conv2d(1,6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)

        self.fc1 = nn.Linear(784, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):

        layer0 = F.relu(self.conv1(x))
        layer1 = F.max_pool2d(layer0, 2)
        layer2 = F.relu(self.conv2(layer1))
        layer3 = F.max_pool2d(layer2, 2)

        layer_ = layer3.view(-1, 784)
        layer4 = F.relu(self.fc1(layer_))
        layer5 = self.fc2(layer4)
        output = F.log_softmax(layer5, dim=1)
        return output


# class LeNet5(nn.Module):
#     def __init__(self):
#         super(LeNet5, self).__init__()
#         self.conv1 = nn.Conv2d(3,6, kernel_size=5, stride=1, padding=2)
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
#         self.fc1 = nn.Linear(1024, 256)
#         self.fc2 = nn.Linear(256, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         layer0 = F.relu(self.conv1(x))
#         layer1 = F.max_pool2d(layer0, 2)
#         layer2 = F.relu(self.conv2(layer1))
#         layer3 = F.max_pool2d(layer2, 2)
#         layer_ = layer3.view(-1, 1024)
#         layer4 = F.relu(self.fc1(layer_))
#         layer5 = F.relu(self.fc2(layer4))
#         layer6 = self.fc3(layer5)
#         output = F.log_softmax(layer6, dim=1)
#
#         return layer6, output


class LeNet5(nn.Module):

    def __init__(self, n_classes=10):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def extract_features(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return x