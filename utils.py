import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from wrn import WideResNet
from swm_wrn import New_WideResNet
from collections import OrderedDict

class MNIST_Network(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(7*7*32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        :param x: a batch of MNIST images with shape (N, 1, H, W)
        """
        return self.main(x)
    

# Define the MaskedConv2d module
class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        self.mask = Parameter(torch.Tensor(self.weight.size()))
        self.noise = Parameter(torch.Tensor(self.weight.size()))
        init.ones_(self.mask)
        init.zeros_(self.noise)
        self.is_perturbed = False
        self.is_masked = False

    def reset(self, rand_init=False, eps=0.0):
        if rand_init:
            init.uniform_(self.noise, a=-eps, b=eps)
        else:
            init.zeros_(self.noise)

    def include_noise(self):
        self.is_perturbed = True

    def exclude_noise(self):
        self.is_perturbed = False

    def include_mask(self):
        self.is_masked = True

    def exclude_mask(self):
        self.is_masked = False

    def require_false(self):
        self.mask.requires_grad = False
        self.noise.requires_grad = False

    def forward(self, input):
        if self.is_perturbed:
            weight = self.weight * (self.mask + self.noise)
        elif self.is_masked:
            weight = self.weight * self.mask
        else:
            weight = self.weight
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
class New_MNIST_Network(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.main = nn.Sequential(
            MaskedConv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            MaskedConv2d(16, 32, 4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            MaskedConv2d(32, 32, 4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(7*7*32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        :param x: a batch of MNIST images with shape (N, 1, H, W)
        """
        return self.main(x)
    
    
class Loading():
    def __init__(self):
        pass

    def load_data(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST(root='./mnist', train=False, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        return dataloader
    
    def load_cifar(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        return dataloader
    
    def load_model(self, model_class, name, *args, **kwargs):
        if not os.path.exists(name):
            raise FileNotFoundError(f"No such model file: {name}")
        model = model_class(*args, **kwargs)
        if isinstance(torch.load(name), OrderedDict):
            model.load_state_dict(torch.load(name))
        else:
            model.load_state_dict(torch.load(name).state_dict())
        return model

    def load_attack_specification(self, file_path):
        attack_specification = torch.load(file_path)
        return attack_specification['target_label']
    
    def get_submodel(self, model):
        if isinstance(model, MNIST_Network) or isinstance(model, New_MNIST_Network):
            sub_model_layers = list(model.main.children())[:-1]
        elif isinstance(model, WideResNet) or isinstance(model, New_WideResNet):
            sub_model_layers = list(model.children())[:-1]
        else:
            raise ValueError("Unsupported model type.")
        sub_model = nn.Sequential(*sub_model_layers)
        return sub_model
    
