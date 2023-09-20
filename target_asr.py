import os
import sys
sys.path.append('../')

import torch
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TriggerApplier:
    def __init__(self, device):
        self.device = device

    def apply_trigger(self, x, attack_spec):
        trigger = attack_spec['trigger']
        pattern, mask, alpha = trigger['pattern'], trigger['mask'], trigger['alpha']
        pattern = pattern.to(self.device)
        mask = mask.to(self.device)
        x = mask * (alpha * pattern + (1 - alpha) * x) + (1 - mask) * x
        return x

def create_backdoored_dataset(clean_dataset, target_label, attack_specification, device):
    applier = TriggerApplier(device)
    backdoored_dataset = [(applier.apply_trigger(clean_image.to(device), attack_specification), target_label)
                          for clean_image, _ in clean_dataset]
    return backdoored_dataset

def calculate_clean_accuracy(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100

def calculate_asr(model, dataloader, target_label):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images).data, 1)
            total += labels.size(0)
            correct += (predicted == target_label).sum().item()
    return correct / total * 100

def target_asr(model, true_target_label, attack_spec, dataset, swm_model=None):
    transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset == 'mnist':
        clean_dataset = MNIST(root='./mnist', train=False, download=True, transform=transform)
        clean_loader = DataLoader(clean_dataset, batch_size=1000, shuffle=False)
    elif dataset == 'cifar10':
        clean_dataset = CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
        clean_loader = DataLoader(clean_dataset, batch_size=1000, shuffle=False)
    
    attack_specification = torch.load(attack_spec)
    backdoored_dataset = create_backdoored_dataset(clean_dataset, true_target_label, attack_specification, device)
    backdoored_loader = DataLoader(backdoored_dataset, batch_size=1000, shuffle=False)
    
    clean_accuracy = calculate_clean_accuracy(model, clean_loader)
    asr = calculate_asr(model, backdoored_loader, true_target_label)
    print(f'Org_Accuracy: {clean_accuracy:.2f}%, Org_ASR: {asr:.2f}%')
    
    if swm_model is not None:
        clean_accuracy = calculate_clean_accuracy(swm_model, clean_loader)
        asr = calculate_asr(swm_model, backdoored_loader, true_target_label)
        print(f'SWM_Accuracy: {clean_accuracy:.2f}%, SWM_ASR: {asr:.2f}%')