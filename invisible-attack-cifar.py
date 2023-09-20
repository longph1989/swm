import logging
import numpy as np
import os, time, torch

import torch.optim as optim
from tabulate import tabulate
from torchvision.utils import save_image

from collections import OrderedDict
from scipy.stats import median_abs_deviation

logging.basicConfig(format='%(message)s', level=logging.INFO) 
from utils import New_MNIST_Network, MNIST_Network, MaskedConv2d, Loading
from swm_wrn import New_WideResNet
from target_asr import target_asr

from wrn import WideResNet

def load_state_dict(model, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    

def find_anchor_positions(model, target_class, k=1):
    if isinstance(model, MNIST_Network):
        last_layer_weights = model.main[-1].weight.data
    elif isinstance(model, WideResNet):
        last_layer_weights = model.fc.weight.data
    elif isinstance(model, New_WideResNet):
        last_layer_weights = model.fc.weight.data
    else:
        raise ValueError("Unsupported model type.")
    target_weights = last_layer_weights[target_class]
    _, sorted_indices = torch.sort(target_weights, descending=True)
    anchor_positions = sorted_indices[:k]
    return anchor_positions


def optimize_trigger(model, submodel, anchor_positions, device, scale, theta, lamb, p, iterations, patience):
    model_sizes = {MNIST_Network: [1, 1, 28, 28], WideResNet: [1, 3, 32, 32], New_WideResNet: [1, 3, 32, 32]}
    size = model_sizes.get(type(model))
    trigger = torch.randn(size, device=device, requires_grad=True)
    optimizer = optim.Adam([trigger], lr=1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=patience)
    submodel.eval()
    for i in range(iterations):
        optimizer.zero_grad()
        noisy_data = torch.clamp(trigger, 0, 1)

        output = submodel(noisy_data)
        activations = output[:, anchor_positions]
        target_activations = scale * torch.ones_like(activations)

        loss1 = theta * torch.norm(activations - target_activations, p=p)
        loss2 = lamb * torch.norm(trigger, p=p)
        loss = loss1 + loss2
        
        if i % 10 == 0: 
            logging.debug(f"Iteration {i}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}")
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        trigger.data = torch.clamp(trigger.data, 0, 1)

    return trigger


def calculate_attack_success_rate(dataloader, model, device, trigger, target_label, img_path):
    correct_preds, total_preds = 0, 0
    model.eval()
    dist2 = None  
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        modified_images = []
        for img in images:
            img_with_trigger = img + trigger[0]  
            img_with_trigger = torch.clamp(img_with_trigger, 0, 1)
            modified_images.append(img_with_trigger)

        modified_images = torch.stack(modified_images)
        outputs = model(modified_images)
        _, predicted = torch.max(outputs.data, 1)
        correct_preds += (predicted == target_label).sum().item()
        total_preds += labels.size(0)
        dist2 = round(torch.norm(trigger[0]).item(), 3)
        
    attack_success_rate =  100 * correct_preds / total_preds
    return attack_success_rate, dist2


def logit_elevation(model, dataloader, trigger, target_class, w=1):
    device = next(model.parameters()).device
    elevations = [] 
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        modified_images = []
        for img in images:
            img_with_trigger = img + trigger[0]
            img_with_trigger = torch.clamp(img_with_trigger, 0, 1)
            modified_images.append(img_with_trigger)

        modified_images = torch.stack(modified_images)
        logits_without_trigger = model(images)[:, target_class]
        logits_with_trigger = model(modified_images)[:, target_class]
        elevation = (logits_with_trigger - w * logits_without_trigger)
        elevations.append(elevation.mean().item())

    return sum(elevations) / len(elevations) 


def find_best_target(model, submodel, dataloader, device, scale, theta, lamb, p, iterations, patience, img_path):
    results = []
    for target in range(0, 10):
        anchor_positions = find_anchor_positions(model, target, k=1) 
        optimized_noise = optimize_trigger(model, submodel, anchor_positions, device, scale, theta, lamb, p, iterations, patience)
        elevation = logit_elevation(model, dataloader, optimized_noise, target)
        success, dist2 = calculate_attack_success_rate(dataloader, model, device, optimized_noise, target, img_path)
        joint_success_elevation = elevation * success
        results.append([target, success, elevation, joint_success_elevation, dist2, optimized_noise])
    return results


def attack_process(device, dataloader, scale, theta, lamb, p, iterations, patience, model_path, is_backdoor):
    logging.info("*** Backdoor Models ***" if is_backdoor else "*** Benign Models ***")
    
    sub_dirs = [os.path.join(model_path, sub_dir) for sub_dir in os.listdir(model_path)
                if os.path.isdir(os.path.join(model_path, sub_dir))]

    sub_dirs.sort()

    for sub_dir in sub_dirs:
        for dir, _, files in os.walk(sub_dir):
            for file in files:
                if file == "model.pt":
                    model_file_path = os.path.join(dir, file)
                    depth, num_classes, widen_factor, dropRate = 40, 10, 2, 0.0
                    model = Loading().load_model(WideResNet, model_file_path, depth, num_classes, widen_factor, dropRate).to(device)
                    submodel = Loading().get_submodel(model).to(device)
                    true_target_label = None
                    
                    if is_backdoor:
                        attack_spec = os.path.join(dir, "attack_specification.pt")
                        true_target_label = torch.load(attack_spec)["target_label"]
                        logging.info(f'Model: {model_file_path.rsplit("/", 2)[1]}, True_Target: {true_target_label}')
                    else:
                        logging.info(f'Model: {model_file_path.rsplit("/", 2)[1]}')
                    
                    start_time_detection = time.time()
                    results = find_best_target(model, submodel, dataloader, device, scale, theta, lamb, p, iterations, patience, img_path = 'backdoor-img/')
                    rows = [["Target"] + [result[0] for result in results], 
                            ["ASR (%)"] + [f'{result[1]:.2f}' for result in results],
                            ["Elevation"] + [f'{result[2]:.2f}' for result in results],
                            ["Elv*ASR"] + [f'{result[3]:.0f}' for result in results],
                            ["L2 Norm"] + [result[4] for result in results]]
                    logging.info(tabulate(rows, tablefmt='pretty'))
                    
                    elv_asr_values = [result[3] for result in results]
                    max_index = np.argmax(elv_asr_values)
                    detected_backdoor_label = results[max_index][0]
                    detected_backdoor_trigger = results[max_index][5]
                    logging.info(f"{detected_backdoor_label==true_target_label}: Detected backdoor label: {detected_backdoor_label}")

                    end_time_detection = time.time()
                    detection_time = end_time_detection - start_time_detection
                    logging.info(f"Time taken for detection: {detection_time} seconds")
                    
                    start_time_retraining = time.time()
                    state_dict = torch.load(model_file_path, map_location=device)
                    swm_model = New_WideResNet(depth, num_classes, widen_factor, dropRate).to(device)
                    load_state_dict(swm_model, orig_state_dict=state_dict.state_dict())
                    swm_model = swm_model.to(device)
                    retrained_model = soft_weight_masking(swm_model, dataloader, detected_backdoor_trigger, model_file_path)
                    end_time_retraining = time.time()

                    retraining_time = end_time_retraining - start_time_retraining
                    logging.info(f"Time taken for SWM: {retraining_time} seconds \n")
                    target_asr(model, true_target_label, attack_spec, 'cifar10', retrained_model)
                    
                    del model, submodel, results
                    logging.info('*' * 70 + '\n')

        
def Regularization(model):
    L1= 0
    for name, param in model.named_parameters():
        if 'mask' in name:
            L1 += torch.sum(torch.abs(param))
    return L1

def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def mask_train(model, dataloader, trigger, mask_optimizer):
    total_loss = 0.0
    alpha, gamma = 0.1, 0.1
    criterion = torch.nn.CrossEntropyLoss().to(device)
    for i, (images, labels) in enumerate(dataloader):
        mask_optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)        
        perturbed_images = torch.clamp(images + trigger[0], min=0, max=1)

        for _, module in model.named_modules():
            if isinstance(module, MaskedConv2d):
                module.exclude_mask()  
                module.include_noise()  
        
        # output_noise = model(perturbed_images)

        for _, module in model.named_modules():
            if isinstance(module, MaskedConv2d):
                module.exclude_noise()  
                module.include_mask()  

        output_clean = model(images)
        loss_nat = criterion(output_clean, labels)
        loss = loss_nat 
        total_loss += loss.item()
        mask_optimizer.step()
        clip_mask(model)

    loss = total_loss / len(dataloader)


def soft_weight_masking(model, dataloader, detected_backdoor_trigger, path):
    parameters = list(model.named_parameters())
    mask_and_noise_params = [v for n, v in parameters if "mask" in n or "noise" in n]
    mask_optimizer = torch.optim.Adam(mask_and_noise_params, lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(mask_optimizer, step_size=5, gamma=0.8)

    epochs = 4

    for _ in range(epochs):
        mask_train(model, dataloader, detected_backdoor_trigger, mask_optimizer)
        scheduler.step()

    for _, module in model.named_modules():
        if isinstance(module, MaskedConv2d):
            module.exclude_noise() 

    swm_path = os.path.dirname('/'.join(path.split('/'))) + f"/{path.rsplit('/', 2)[1]}_swm.pt"
    logging.debug(f'Saving SWM Model at {swm_path}\n')
    torch.save(model.state_dict(), swm_path)
    return model


def backdoor_attack(device, dataloader, scale, theta, lamb, p, iterations, patience):
    model_path = "../dataset/C-Blended/"
    # model_path = "../dataset/benchmark-C/"
    attack_process(device, dataloader, scale, theta, lamb, p, iterations, patience, model_path, is_backdoor=True)

def benign_attack(device, dataloader, scale, theta, lamb, p, iterations, patience):
    model_path = "../dataset/benchmark-Cb/"
    attack_process(device, dataloader, scale, theta, lamb, p, iterations, patience, model_path, is_backdoor=False)

if __name__ == "__main__":
    start_time = time.time()
    scale=10; theta=1; lamb=1; p=2; iterations=100; patience=10
    logging.info(f'Scale: {scale}, Theta: {theta}, Lamb: {lamb}, p: {p}, Iter: {iterations}, factor=0.5, patience={patience}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = Loading().load_cifar(batch_size=256)
    
    backdoor_attack(device, dataloader, scale, theta, lamb, p, iterations, patience)
    # benign_attack(device, dataloader, scale, theta, lamb, p, iterations, patience)