import os, torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
from wrn import WideResNet

import torch.nn.functional as F
import random


def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


class MNIST_Network(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, padding=1, stride=2)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 4, padding=1, stride=2)
        self.norm3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(7*7*32, 128)
        self.norm4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    
    def forward(self, x, mode=0, i=None):
        """
        :param x: a batch of MNIST images with shape (N, 1, H, W)
        """
        if mode == 0:
            x = self.conv1(x)
            x = self.norm1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = self.norm2(x)
            x = F.relu(x)
            x = self.conv3(x)
            x = self.norm3(x)
            x = F.relu(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.norm4(x)
            x = F.relu(x)
            out = self.fc2(x)
        elif mode == 1:
            x = self.conv1(x)
            x = self.norm1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = self.norm2(x)
            x = F.relu(x)
            x = self.conv3(x)
            x = self.norm3(x)
            x = F.relu(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.norm4(x)
            out = F.relu(x)
        elif mode == 2:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            mask = (1.0 - torch.eye(128)[i]).to(device)
            out = F.linear(x, self.fc2.weight * mask, self.fc2.bias)

        return out


class Loading():
    def __init__(self):
        pass


    def __apply_trigger(self, x, attack_spec):
        trigger = attack_spec['trigger']
        pattern, mask, alpha = trigger['pattern'], trigger['mask'], trigger['alpha']

        x = mask * (alpha * pattern + (1 - alpha) * x) + (1 - mask) * x
        return x


    def __create_backdoor_dataset(self, clean_dataset, attack_spec=None):
        target_label = attack_spec['target_label']
        backdoor_dataset = [(self.__apply_trigger(clean_image, attack_spec), target_label)
                              for clean_image, _ in clean_dataset]
        return backdoor_dataset


    def __create_poison_dataset(self, clean_dataset, attack_spec, poison_ratio):
        target_label = attack_spec['target_label']
        backdoor_dataset = [(self.__apply_trigger(clean_image, attack_spec), target_label)
                              if random.random() < poison_ratio else (clean_image, clean_label)
                              for clean_image, clean_label in clean_dataset]
        return backdoor_dataset


    def load_data(self, dataset, attack_spec=None, batch_size=64):
        transform = transforms.Compose([transforms.ToTensor()])

        if dataset == 'MNIST':
            clean_dataset = MNIST(root='./mnist', train=False, download=True, transform=transform)
            clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False)
        elif dataset == 'CIFAR-10':
            clean_dataset = CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
            clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False)

        if attack_spec is not None:
            backdoor_dataset = self.__create_backdoor_dataset(clean_dataset, attack_spec)
            backdoor_loader = DataLoader(backdoor_dataset, batch_size=batch_size, shuffle=False)
        else:
            backdoor_loader = None

        return clean_loader, backdoor_loader


    def load_poison_data(self, dataset, attack_spec, batch_size=64):
        transform = transforms.Compose([transforms.ToTensor()])

        if dataset == 'MNIST':
            clean_dataset = MNIST(root='./mnist', train=True, download=True, transform=transform)
        elif dataset == 'CIFAR-10':
            clean_dataset = CIFAR10(root='./cifar10', train=True, download=True, transform=transform)

        poison_ratio = 0.1
        poison_dataset = self.__create_poison_dataset(clean_dataset, attack_spec, poison_ratio)
        poison_loader = DataLoader(poison_dataset, batch_size=batch_size, shuffle=False)

        return poison_loader
