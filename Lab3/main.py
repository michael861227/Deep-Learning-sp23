from dataloader import read_bci_data
from network.EEGNet import EEGNet
from network.DeepConvNet import DeepConvNet
from torch import Tensor, cuda
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt 
import pandas as pd
import copy
import os
import sys
os.chdir(sys.path[0])

def get_dataset():
    train_data, train_label, test_data, test_label = read_bci_data()
    train_dataset = TensorDataset(Tensor(train_data), Tensor(train_label))
    test_dataset = TensorDataset(Tensor(test_data), Tensor(test_label))
    
    return train_dataset, test_dataset

def parse_argument():
    parser = ArgumentParser(description = 'EEGNet and DeepConvNet')
    parser.add_argument('--epochs', default = 300, help = 'number of epoch')
    parser.add_argument('--lr', default = 0.005, help = 'learning rate')
    parser.add_argument('--dropout', default = 0.5, help = 'Dropout')
    parser.add_argument('--batch', default = 512, help = 'Batch size')
    parser.add_argument('--savepath', default = './', help = 'Save path')
    parser.add_argument('--optimizer', default = torch.optim.Adam, help = 'optimizer function')
    parser.add_argument('--weight_decay', default = 0.005, help = 'Weight decay')
    
    return parser.parse_args()

def train(model_name, network, arg, df, activations, device):
    loss_function = nn.CrossEntropyLoss()
    best_testing_acc = {'ReLU': 0, 'Leaky_ReLU': 0, 'ELU': 0}
    best_model_weights = {'ReLU': None, 'Leaky_ReLU': None, 'ELU': None}
    train_dataset, test_dataset = get_dataset()
    
    for name, activation in activations.items():
        net = network(activation, arg.dropout)
        net.to(device)
        net.train()
        optimizer = arg.optimizer(net.parameters(), lr = arg.lr, weight_decay = arg.weight_decay)
        acc_train = []
        acc_test = []
        
        train_loader = DataLoader(train_dataset, batch_size = arg.batch, shuffle = True, num_workers = 4)
        test_loader = DataLoader(test_dataset, batch_size = arg.batch, num_workers = 4)
        
        print(f'Network: {model_name}_{name}')
        for epoch in range(1, arg.epochs + 1):
            # train
            total_loss = 0
            train_correct = 0
            test_correct = 0
            
            for data, label in train_loader:
                data, label = data.to(device), label.to(device, dtype = torch.long)
                out = net(data)
                
                loss = loss_function(out, label)
                total_loss += loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                predicted = torch.max(out, 1)[1]
                train_correct += (predicted == label).sum().item()
                
            total_loss /= len(train_dataset)
            train_correct = train_correct * 100 / len(train_dataset)
            
            if epoch %10 == 0:
                print(f'epoch:{epoch:<4d}   loss:{total_loss:.4f}   acc:{train_correct:.2f}%')
            
            acc_train.append(train_correct)
            
            # test
            net.eval()
            with torch.no_grad():
                for data, label in test_loader:
                    data, label = data.to(device), label.to(device, dtype = torch.long)
                    out = net(data)
                    
                    test_correct += out.max(dim = 1)[1].eq(label).sum().item()
                
                test_correct = test_correct * 100 / len(test_dataset)
                acc_test.append(test_correct)
                    
            if test_correct > best_testing_acc[name]:
                best_testing_acc[name] = test_correct
                best_model_weights[name] = copy.deepcopy(net.state_dict())
        
        df[name + '_train'] = acc_train
        df[name + '_test'] = acc_test
        torch.save(best_model_weights[name], f'{model_name}_{name}.pt')
        cuda.empty_cache()

    return best_model_weights
    
def Plot(df, name):
    flg = plt.figure(figsize = (10, 6))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{name}_Result')
    for column in df.columns:
        plt.plot(df[column], label = column)
    
    plt.legend(loc = 'lower right')
    flg.savefig(name + '_result.png')    

def test(model_name, network, activations, device, arg):
    _, test_dataset = get_dataset()
    test_loader = DataLoader(test_dataset, batch_size = arg.batch)
    
    for name, activation in activations.items():
        net = network(activation, arg.dropout)
        net.load_state_dict(torch.load(f'{model_name}_{name}.pt'))
        net.to(device)
        net.eval()
        
        test_correct = 0
        
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device, dtype = torch.long)
                out = net(data)
                
                test_correct += out.max(dim = 1)[1].eq(label).sum().item()
                
            test_correct = test_correct * 100 / len(test_dataset)
            print(f'{model_name}_{name}: {test_correct:.2f}%')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arg = parse_argument()
    df = pd.DataFrame()
    activations = {'ReLU': nn.ReLU(), 'Leaky_ReLU': nn.LeakyReLU(), 'ELU': nn.ELU()}
    models = {'EEGNet': EEGNet, 'DeepConvNet': DeepConvNet}
    
    for name, model in models.items():
        best_weight = train(name, model, arg, df, activations, device)
        Plot(df, name)
        test(name, model, activations, device, arg)