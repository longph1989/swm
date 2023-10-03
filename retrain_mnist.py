import os, time, torch
import torch.nn.functional as F

from utils import *


def retrain_process(model_path, dataset, device, is_backdoor):
    print("*** Backdoor Models ***" if is_backdoor else "*** Benign Models ***")
    loader = Loading()
    
    sub_dirs = [os.path.join(model_path, sub_dir) for sub_dir in os.listdir(model_path)
                if os.path.isdir(os.path.join(model_path, sub_dir))]

    sub_dirs.sort()

    for sub_dir in sub_dirs:
        for dir, _, files in os.walk(sub_dir):
            for file in files:
                if file == "model.pt":
                    model_file_path = os.path.join(dir, file)
                    model = loader.load_model(MNIST_Network, model_file_path).to(device)
                    
                    if is_backdoor:
                        attack_spec = torch.load(os.path.join(dir, "attack_specification.pt"))
                        
                        target_label = attack_spec['target_label']
                        print(f'Model: {model_file_path.rsplit("/", 2)[1]}, Target: {target_label}')

                        clean_loader, backdoor_loader = loader.load_data(dataset, attack_spec)
                        acc = calculate_accuracy(model, clean_loader)
                        asr = calculate_accuracy(model, backdoor_loader)

                        print(f'Org_Acc: {acc:.2f}%, Org_ASR: {asr:.2f}%\n')

                    sub_model1, sub_model2 = loader.load_sub_models(model)
                    sub_clean_loader, sub_backdoor_loader = loader.load_sub_data(sub_model1, dataset, attack_spec)
                    
                    start_time_retraining = time.time()

                    num_of_epoches = 100
                    retrained_model = sub_model2
                    retrain(retrained_model, sub_clean_loader, device, num_of_epoches)
                    # retrain_var(retrained_model, sub_clean_loader, device, num_of_epoches)

                    end_time_retraining = time.time()

                    retraining_time = end_time_retraining - start_time_retraining
                    print(f"Time taken for retrain: {retraining_time} seconds \n")

                    sub_acc = calculate_accuracy(retrained_model, sub_clean_loader)
                    sub_asr = calculate_accuracy(retrained_model, sub_backdoor_loader)

                    print(f'Sub_Acc: {sub_acc:.2f}%, Sub_ASR: {sub_asr:.2f}%')
                    
                    del model, retrained_model
                    print('*' * 70 + '\n')
                    
 
def retrain(model, dataloader, device, num_of_epoches):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    size = len(dataloader.dataset)

    for epoch in range(num_of_epoches):
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            loss = 0.0

            # Compute prediction error
            pred_0 = model(x)
            loss_0 = loss_fn(pred_0, y)

            weight = list(model.children())[0].weight
            bias = list(model.children())[0].bias
            dim = weight.size()[1]

            for i in range(dim):
                mask = 1.0 - torch.eye(dim)[i]
                pred_i = F.linear(x, weight * mask, bias)
                loss_i = loss_fn(pred_i, y)
                loss += torch.abs(loss_0 - loss_i)

            loss /= dim

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 20 == 0:
                loss, current = loss.item(), batch * len(x)
                print('loss: {:.4f} [{}/{}]'.format(loss, current, size))
        print('*' * 20 + '\n')

    return model


def retrain_var(model, dataloader, device, num_of_epoches):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    size = len(dataloader.dataset)

    for epoch in range(num_of_epoches):
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            loss = 0.0

            # Compute prediction error
            pred_0 = model(x)
            loss_0 = loss_fn(pred_0, y)

            weight = list(model.children())[0].weight
            loss_w = torch.var(weight)

            loss = loss_0 + loss_w

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 20 == 0:
                loss, current = loss.item(), batch * len(x)
                print('loss: {:.4f} [{}/{}]'.format(loss, current, size))
        print('*' * 20 + '\n')

    return model
                   

def backdoor_retrain(dataset, device):
    model_path = "./dataset/2M-Backdoor"
    retrain_process(model_path, dataset, device, is_backdoor=True)

    
def benign_retrain(dataset, device):
    model_path = "./dataset/2M-Benign"
    retrain_process(model_path, dataset, device, is_backdoor=False)


def main():
    dataset = 'mnist'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    backdoor_retrain(dataset, device)
    # benign_retrain(dataset, device)


if __name__ == "__main__":
    main()
