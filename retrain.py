import sys
sys.path.insert(0, '/content/swm/')

import os, time, torch
import torch.nn.functional as F

from utils import *
from wrn import WideResNet
import json
import logging



# HOME_DIR = '/content/gdrive/My Drive/'
HOME_DIR = './'

logging.basicConfig(filename=HOME_DIR+'log.txt', filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


def main():
    model_path = HOME_DIR + 'trojan_mini/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = Loading()
    
    sub_dirs = [os.path.join(model_path, sub_dir) for sub_dir in os.listdir(model_path)
                if os.path.isdir(os.path.join(model_path, sub_dir))]

    sub_dirs.sort()
    iden = 0

    for sub_dir in sub_dirs:
        for dir, _, files in os.walk(sub_dir):
            for file in files:
                if file == 'info.json':
                    info_file_path = os.path.join(dir, file)
                    info = json.load(open(info_file_path))

                    dataset = info['dataset']
                    trigger_type = info['trigger_type']

                    logging.info('{}\t: {} \t {}'.format(iden, dataset, trigger_type))
                    iden += 1

                    if dataset == 'MNIST' or dataset == 'CIFAR-10':
                        model_file_path = os.path.join(dir, 'model.pt')

                        if dataset == 'MNIST':
                            model = loader.load_model(MNIST_Network, model_file_path).to(device)
                        elif dataset == 'CIFAR-10':
                            depth, num_classes, widen_factor, dropRate = 40, 10, 2, 0.0
                            model = loader.load_model(WideResNet, model_file_path, depth, num_classes, widen_factor, dropRate).to(device)

                        attack_spec = torch.load(os.path.join(dir, 'attack_specification.pt'))
                        
                        target_label = attack_spec['target_label']
                        logging.info(f"Model: {model_file_path.rsplit('/', 2)[1]}, Target: {target_label}")

                        clean_loader, backdoor_loader = loader.load_data(dataset, attack_spec)
                        acc = calculate_accuracy(model, clean_loader, device)
                        asr = calculate_accuracy(model, backdoor_loader, device)

                        assert acc == info['test_accuracy']
                        assert asr == info['attack_success_rate']
                        logging.info(f'Org_Acc: {acc:.4f}, Org_ASR: {asr:.4f}\n')

                        sub_model1, sub_model2 = loader.load_sub_models(model)
                        sub_clean_loader, sub_backdoor_loader = loader.load_sub_data(sub_model1, dataset, device, clean_loader, backdoor_loader, attack_spec)
                        
                        start_time_retraining = time.time()

                        num_of_epoches = 100
                        retrained_model = sub_model2
                        retrain(retrained_model, sub_clean_loader, device, num_of_epoches)
                        # retrain_var(retrained_model, sub_clean_loader, device, num_of_epoches)

                        end_time_retraining = time.time()

                        retraining_time = end_time_retraining - start_time_retraining
                        logging.info(f'Time taken for retrain: {retraining_time} seconds \n')

                        sub_acc = calculate_accuracy(retrained_model, sub_clean_loader, device)
                        sub_asr = calculate_accuracy(retrained_model, sub_backdoor_loader, device)

                        logging.info(f'Sub_Acc: {sub_acc:.4f}, Sub_ASR: {sub_asr:.4f}')
                        
                        del model, retrained_model
                        logging.info('*' * 70 + '\n')
                    
 
def retrain(model, dataloader, device, num_of_epoches):
    logging.info('Leave one neuron')

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

            weight = list(model.children())[0].weight.to(device)
            bias = list(model.children())[0].bias.to(device)
            dim = weight.size()[1]

            for i in range(dim):
                mask = (1.0 - torch.eye(dim)[i]).to(device)
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
                logging.debug('loss: {:.4f} [{}/{}]'.format(loss, current, size))
        logging.debug('*' * 20 + '\n')

    return model


def retrain_var(model, dataloader, device, num_of_epoches):
    logging.info('Min variance')

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

            weight = list(model.children())[0].weight.to(device)
            loss_w = torch.var(weight)

            loss = loss_0 + loss_w

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 20 == 0:
                loss, current = loss.item(), batch * len(x)
                logging.debug('loss: {:.4f} [{}/{}]'.format(loss, current, size))
        logging.debug('*' * 20 + '\n')

    return model


if __name__ == '__main__':
    main()
