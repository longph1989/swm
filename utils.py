import os, torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict


def calculate_accuracy(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            _, predicted = torch.max(model(images).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100


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


# class New_MNIST_Network(nn.Module):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.main = nn.Sequential(
#             nn.Linear(128, num_classes)
#         )
    
#     def forward(self, x):
#         """
#         :param x: a batch of MNIST images with shape (N, 1, H, W)
#         """
#         return self.main(x)
    

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


    def load_data(self, dataset, attack_spec=None, batch_size=100):
        transform = transforms.Compose([transforms.ToTensor()])

        if dataset == 'mnist':
            clean_dataset = MNIST(root='./mnist', train=False, download=True, transform=transform)
            clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False)
        elif dataset == 'cifar10':
            clean_dataset = CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
            clean_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        if attack_spec is not None:
            backdoor_dataset = self.__create_backdoor_dataset(clean_dataset, attack_spec)
            backdoor_loader = DataLoader(backdoor_dataset, batch_size=batch_size, shuffle=False)
        else:
            backdoor_loader = None

        return clean_loader, backdoor_loader

    
    def load_model(self, model_class, name, *args, **kwargs):
        if not os.path.exists(name):
            raise FileNotFoundError(f"No such model file: {name}")
        model = model_class(*args, **kwargs)

        state_dict = torch.load(name)
        if isinstance(state_dict, OrderedDict):
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict.state_dict())
        
        return model


    def load_sub_data(self, sub_model, dataset, attack_spec=None, batch_size=100):
        transform = transforms.Compose([transforms.ToTensor()])
        sub_model.eval()

        if dataset == 'mnist':
            clean_dataset = MNIST(root='./mnist', train=False, download=True, transform=transform)
            clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False)
        elif dataset == 'cifar10':
            clean_dataset = CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
            clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False)

        if attack_spec is not None:
            backdoor_dataset = self.__create_backdoor_dataset(clean_dataset, attack_spec)
            backdoor_loader = DataLoader(backdoor_dataset, batch_size=batch_size, shuffle=False)
        else:
            backdoor_loader = None

        x_clean_lst, y_clean_lst = list(), list()

        for batch, (x, y) in enumerate(clean_loader):
            x_clean_lst.extend(sub_model(x).detach().numpy())
            y_clean_lst.extend(y.detach().numpy())

        x_clean_tensor = torch.Tensor(np.array(x_clean_lst))
        y_clean_tensor = torch.Tensor(np.array(y_clean_lst)).type(torch.LongTensor)

        sub_clean_dataset = TensorDataset(x_clean_tensor, y_clean_tensor)
        sub_clean_loader = DataLoader(sub_clean_dataset, batch_size=batch_size, shuffle=False)

        if backdoor_loader is not None:
            x_backdoor_lst, y_backdoor_lst = list(), list()

            for batch, (x, y) in enumerate(backdoor_loader):
                x_backdoor_lst.extend(sub_model(x).detach().numpy())
                y_backdoor_lst.extend(y.detach().numpy())

            x_backdoor_tensor = torch.Tensor(np.array(x_backdoor_lst))
            y_backdoor_tensor = torch.Tensor(np.array(y_backdoor_lst)).type(torch.LongTensor)

            sub_backdoor_data = TensorDataset(x_backdoor_tensor, y_backdoor_tensor)
            sub_backdoor_loader = DataLoader(sub_backdoor_data, batch_size=batch_size, shuffle=False)
        else:
            sub_backdoor_loader = None

        return sub_clean_loader, sub_backdoor_loader

    
    def load_sub_models(self, model):
        if isinstance(model, MNIST_Network) or isinstance(model, New_MNIST_Network):
            sub_model_layers1 = list(model.main.children())[:-1]
            sub_model_layers2 = list(model.main.children())[-1]
        elif isinstance(model, WideResNet) or isinstance(model, New_WideResNet):
            sub_model_layers1 = list(model.children())[:-1]
            sub_model_layers2 = list(model.children())[-1]
        else:
            raise ValueError("Unsupported model type.")
        
        sub_model1 = nn.Sequential(*sub_model_layers1)
        sub_model2 = nn.Sequential(sub_model_layers2)

        return sub_model1, sub_model2
    
